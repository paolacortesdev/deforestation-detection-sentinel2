import segmentation_models_pytorch as smp


def get_unet_resnet34(
    in_channels=18,
    out_channels=1
):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_channels
    )

    return model