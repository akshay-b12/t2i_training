import io
import os
import random
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from PIL import Image, ImageFile
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torchvision import transforms
from huggingface_hub import login as hf_login

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger("streaming_t2i")
logging.basicConfig(level=logging.INFO)


# -------------------------
# Config
# -------------------------

@dataclass
class DataConfig:
    dataset_name: str = "laion/relaion-pop"
    dataset_split: str = "train"

    image_column_candidates: Tuple[str, ...] = ("image", "jpg", "png")
    caption_column_candidates: Tuple[str, ...] = (
        "text",
        "caption",
        "prompt",
        "caption_long_llama32",
        "caption_long",
        "alt_txt",
    )

    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = True
    shuffle_buffer: int = 10_000
    seed: int = 42

    train_batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    hf_token: Optional[str] = None
    trust_remote_code: bool = False
    timeout_seconds: int = 10

    # If dataset exposes URLs instead of Image feature, this lets you fetch them.
    allow_url_fetch: bool = True
    user_agent: str = "t2i-distill/1.0"

    # caption hygiene
    min_caption_len: int = 3
    max_caption_len: int = 512

    # distributed
    rank: int = field(default_factory=lambda: int(os.environ.get("RANK", "0")))
    world_size: int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", "1")))


# -------------------------
# HF auth
# -------------------------

def maybe_hf_login(token: Optional[str]) -> None:
    if token is None or token.strip() == "":
        LOGGER.info("No HF token provided. Assuming you already ran `hf auth login` or dataset is accessible.")
        return
    try:
        hf_login(token=token, add_to_git_credential=False)
        LOGGER.info("Logged into Hugging Face Hub.")
    except Exception as e:
        LOGGER.warning(f"HF login failed or not needed: {e}")


# -------------------------
# Image decode helpers
# -------------------------

def pil_from_any(
    image_obj: Any,
    *,
    allow_url_fetch: bool = True,
    timeout_seconds: int = 10,
    user_agent: str = "t2i-distill/1.0",
) -> Optional[Image.Image]:
    try:
        if image_obj is None:
            return None

        if isinstance(image_obj, Image.Image):
            return image_obj.convert("RGB")

        if isinstance(image_obj, bytes):
            return Image.open(io.BytesIO(image_obj)).convert("RGB")

        if isinstance(image_obj, dict):
            # HF Image feature often gives {"bytes":..., "path":...}
            if image_obj.get("bytes", None) is not None:
                return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")

            if image_obj.get("path", None) is not None:
                path = image_obj["path"]
                if isinstance(path, str):
                    if path.startswith("http://") or path.startswith("https://"):
                        if allow_url_fetch:
                            headers = {"User-Agent": user_agent}
                            resp = requests.get(path, timeout=timeout_seconds, headers=headers)
                            resp.raise_for_status()
                            return Image.open(io.BytesIO(resp.content)).convert("RGB")
                        return None
                    return Image.open(path).convert("RGB")

            # fallback common keys
            for k in ("url", "image_url"):
                if k in image_obj and isinstance(image_obj[k], str):
                    if allow_url_fetch:
                        headers = {"User-Agent": user_agent}
                        resp = requests.get(image_obj[k], timeout=timeout_seconds, headers=headers)
                        resp.raise_for_status()
                        return Image.open(io.BytesIO(resp.content)).convert("RGB")
                    return None

        if isinstance(image_obj, str):
            if image_obj.startswith("http://") or image_obj.startswith("https://"):
                if not allow_url_fetch:
                    return None
                headers = {"User-Agent": user_agent}
                resp = requests.get(image_obj, timeout=timeout_seconds, headers=headers)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            return Image.open(image_obj).convert("RGB")

    except Exception:
        return None

    return None


def find_first_present(example: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in example and example[c] is not None:
            return c
    return None


def sanitize_caption(caption: Any, min_len: int, max_len: int) -> Optional[str]:
    if caption is None:
        return None
    if isinstance(caption, list):
        caption = caption[0] if len(caption) > 0 else None
    if caption is None:
        return None
    if not isinstance(caption, str):
        caption = str(caption)
    caption = " ".join(caption.strip().split())
    if len(caption) < min_len:
        return None
    if len(caption) > max_len:
        caption = caption[:max_len]
    return caption


def build_image_transform(resolution: int, center_crop: bool = False, random_flip: bool = True):
    crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)
    ops = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        crop,
    ]
    if random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0,1] -> [-1,1]
        ]
    )
    return transforms.Compose(ops)


# -------------------------
# Streaming iterable dataset
# -------------------------

class StreamingTextImageDataset(IterableDataset):
    def __init__(self, cfg: DataConfig, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.image_transform = build_image_transform(
            resolution=cfg.resolution,
            center_crop=cfg.center_crop,
            random_flip=cfg.random_flip,
        )

        maybe_hf_login(cfg.hf_token)

    def _tokenize_caption(self, caption: str) -> Dict[str, torch.Tensor]:
        toks = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        out = {"input_ids": toks.input_ids[0]}
        if "attention_mask" in toks:
            out["attention_mask"] = toks.attention_mask[0]
        return out

    def _build_stream(self):
        ds = load_dataset(
            self.cfg.dataset_name,
            split=self.cfg.dataset_split,
            streaming=True,
            trust_remote_code=self.cfg.trust_remote_code,
        )

        ds = ds.shuffle(buffer_size=self.cfg.shuffle_buffer, seed=self.cfg.seed)

        # distributed node split
        if self.cfg.world_size > 1:
            ds = split_dataset_by_node(ds, rank=self.cfg.rank, world_size=self.cfg.world_size)

        return ds

    def __iter__(self):
        ds = self._build_stream()

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # Simple modulo split to avoid duplicate iteration across DataLoader workers.
        for idx, example in enumerate(ds):
            if (idx % num_workers) != worker_id:
                continue

            try:
                image_key = find_first_present(example, self.cfg.image_column_candidates)
                caption_key = find_first_present(example, self.cfg.caption_column_candidates)

                if image_key is None or caption_key is None:
                    continue

                image = pil_from_any(
                    example[image_key],
                    allow_url_fetch=self.cfg.allow_url_fetch,
                    timeout_seconds=self.cfg.timeout_seconds,
                    user_agent=self.cfg.user_agent,
                )
                if image is None:
                    continue

                caption = sanitize_caption(
                    example[caption_key],
                    min_len=self.cfg.min_caption_len,
                    max_len=self.cfg.max_caption_len,
                )
                if caption is None:
                    continue

                pixel_values = self.image_transform(image)
                text_tensors = self._tokenize_caption(caption)

                sample = {
                    "pixel_values": pixel_values,
                    "caption": caption,
                    "input_ids": text_tensors["input_ids"],
                }
                if "attention_mask" in text_tensors:
                    sample["attention_mask"] = text_tensors["attention_mask"]

                yield sample

            except Exception:
                continue


def collate_text_image_batch(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = {
        "pixel_values": torch.stack([x["pixel_values"] for x in examples], dim=0).contiguous().float(),
        "input_ids": torch.stack([x["input_ids"] for x in examples], dim=0),
        "captions": [x["caption"] for x in examples],
    }
    if "attention_mask" in examples[0]:
        batch["attention_mask"] = torch.stack([x["attention_mask"] for x in examples], dim=0)
    return batch


def make_streaming_dataloader(cfg: DataConfig, tokenizer) -> DataLoader:
    dataset = StreamingTextImageDataset(cfg, tokenizer)
    return DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        collate_fn=collate_text_image_batch,
        drop_last=True,
    )