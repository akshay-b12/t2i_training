import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


# --------------------------------------------------
# Config
# --------------------------------------------------

@dataclass
class OfflineShardDataConfig:
    metadata_shard_dir: str              # directory containing *.jsonl shards
    image_root: Optional[str] = None     # base path for relative image paths

    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = True

    min_caption_len: int = 1
    max_caption_len: int = 512

    min_side: int = 0
    verify_images_at_init: bool = True
    skip_missing: bool = True
    skip_invalid_json: bool = True

    caption_dropout_prob: float = 0.0

    train_batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    shuffle: bool = True

    # optional controls
    shard_glob: str = "*.jsonl"
    max_shards: Optional[int] = None
    max_samples: Optional[int] = None


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def sanitize_caption(text: Any, min_len: int = 1, max_len: int = 512) -> Optional[str]:
    if text is None:
        return None
    if isinstance(text, list):
        text = text[0] if len(text) > 0 else None
    if text is None:
        return None
    text = str(text)
    text = " ".join(text.strip().split())
    if len(text) < min_len:
        return None
    if len(text) > max_len:
        text = text[:max_len]
    return text


def resolve_relative_image_path(image_path: str, image_root: Optional[str]) -> Path:
    p = Path(image_path)
    if p.is_absolute():
        return p
    if image_root is not None:
        return Path(image_root) / p
    return p


def build_image_transform(resolution: int, center_crop: bool = False, random_flip: bool = True):
    crop = transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution)

    ops = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        crop,
    ]
    if random_flip:
        ops.append(transforms.RandomHorizontalFlip())

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transforms.Compose(ops)


def verify_image_file(path: Path, min_side: int = 0) -> bool:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if min(w, h) < min_side:
                return False
        return True
    except Exception:
        return False
    
class OfflineShardTextImageDataset(Dataset):
    """
    Reads a directory of JSONL metadata shards and combines them into one in-memory sample list.

    Each JSONL row must contain at least:
        - image_path
        - caption
    """

    def __init__(self, cfg: OfflineShardDataConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.transform = build_image_transform(
            resolution=cfg.resolution,
            center_crop=cfg.center_crop,
            random_flip=cfg.random_flip,
        )

        self.samples: List[Dict[str, Any]] = []
        self._load_shards()

    def _tokenize(self, caption: str) -> Dict[str, torch.Tensor]:
        toks = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": toks.input_ids[0],
        }
        if "attention_mask" in toks:
            item["attention_mask"] = toks.attention_mask[0]
        return item

    def _load_shards(self):
        shard_dir = Path(self.cfg.metadata_shard_dir)
        if not shard_dir.exists():
            raise FileNotFoundError(f"metadata_shard_dir not found: {shard_dir}")

        shard_files = sorted(shard_dir.glob(self.cfg.shard_glob))
        if self.cfg.max_shards is not None:
            shard_files = shard_files[: self.cfg.max_shards]

        if len(shard_files) == 0:
            raise RuntimeError(f"No shard files found in {shard_dir} with glob={self.cfg.shard_glob}")

        total_rows = 0
        kept_rows = 0

        for shard_path in shard_files:
            with shard_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if self.cfg.max_samples is not None and kept_rows >= self.cfg.max_samples:
                        break

                    total_rows += 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        row = json.loads(line)
                    except Exception:
                        if self.cfg.skip_invalid_json:
                            continue
                        raise

                    image_path = row.get("image_path", None)
                    caption = row.get("caption", None)

                    if image_path is None:
                        if self.cfg.skip_missing:
                            continue
                        raise ValueError(f"Missing image_path in row from {shard_path}: {row}")

                    caption = sanitize_caption(
                        caption,
                        min_len=self.cfg.min_caption_len,
                        max_len=self.cfg.max_caption_len,
                    )
                    if caption is None:
                        continue

                    resolved_path = resolve_relative_image_path(image_path, self.cfg.image_root)

                    if not resolved_path.exists():
                        if self.cfg.skip_missing:
                            continue
                        raise FileNotFoundError(f"Image not found: {resolved_path}")

                    if self.cfg.verify_images_at_init:
                        if not verify_image_file(resolved_path, min_side=self.cfg.min_side):
                            continue

                    sample = {
                        "image_path": str(resolved_path),
                        "caption": caption,
                        "source_shard": str(shard_path),
                    }

                    # preserve extra metadata if present
                    for k, v in row.items():
                        if k not in sample:
                            sample[k] = v

                    self.samples.append(sample)
                    kept_rows += 1

                if self.cfg.max_samples is not None and kept_rows >= self.cfg.max_samples:
                    break

        print(
            f"[OfflineShardTextImageDataset] "
            f"scanned {len(shard_files)} shards, "
            f"read {total_rows} rows, kept {kept_rows} valid samples."
        )

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples found across metadata shards.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        image_path = sample["image_path"]
        caption = sample["caption"]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            pixel_values = self.transform(img)

        if self.cfg.caption_dropout_prob > 0.0 and torch.rand(1).item() < self.cfg.caption_dropout_prob:
            caption_for_model = ""
        else:
            caption_for_model = caption

        tok = self._tokenize(caption_for_model)

        item = {
            "pixel_values": pixel_values,
            "input_ids": tok["input_ids"],
            "caption": caption,
            "caption_for_model": caption_for_model,
            "image_path": image_path,
            "source_shard": sample["source_shard"],
        }

        if "attention_mask" in tok:
            item["attention_mask"] = tok["attention_mask"]

        return item
    
def collate_offline_shard_batch(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch = {
        "pixel_values": torch.stack([x["pixel_values"] for x in examples], dim=0).contiguous().float(),
        "input_ids": torch.stack([x["input_ids"] for x in examples], dim=0),
        "captions": [x["caption"] for x in examples],
        "captions_for_model": [x["caption_for_model"] for x in examples],
        "image_paths": [x["image_path"] for x in examples],
        "source_shards": [x["source_shard"] for x in examples],
    }

    if "attention_mask" in examples[0]:
        batch["attention_mask"] = torch.stack([x["attention_mask"] for x in examples], dim=0)

    return batch


def make_offline_shard_dataloader(
    cfg: OfflineShardDataConfig,
    tokenizer,
) -> Tuple[OfflineShardTextImageDataset, DataLoader]:
    dataset = OfflineShardTextImageDataset(cfg, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train_batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        collate_fn=collate_offline_shard_batch,
        drop_last=True,
    )

    return dataset, dataloader

'''
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="tokenizer",
)

cfg = OfflineShardDataConfig(
    metadata_shard_dir="/path/to/relaion_pop/metadata_shards",
    image_root="/path/to/relaion_pop/images",
    resolution=512,
    center_crop=False,
    random_flip=True,
    min_caption_len=3,
    max_caption_len=512,
    min_side=512,
    verify_images_at_init=False,   # faster startup while download is incomplete
    skip_missing=True,
    caption_dropout_prob=0.0,
    train_batch_size=8,
    num_workers=8,
    shuffle=True,
)

train_dataset, train_loader = make_offline_shard_dataloader(cfg, tokenizer)

batch = next(iter(train_loader))
print(batch["pixel_values"].shape)
print(batch["captions"][0])
print(batch["image_paths"][0])
print(batch["source_shards"][0])
'''