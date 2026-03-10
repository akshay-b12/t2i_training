def lpips_loss(lpips_model, pred_image, ref_image):
    # expects normalized images in [-1, 1]
    return lpips_model(pred_image, ref_image).mean()