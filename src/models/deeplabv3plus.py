import segmentation_models_pytorch as smp


def get_deeplabv3plus(
    in_channels=18,
    out_channels=1
):

    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=out_channels
    )

    return model