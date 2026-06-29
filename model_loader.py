from src.models.nnunet_classifier import NNUNetClassifier
from src.models.swin_unetr_classifier import SwinUNETRClassifier
from src.models.efficientnet_classifier import EfficientNetB0Classifier


def load_cls_model(model_name, in_channels=3, num_classes=5, img_size=224):

    if model_name == "nnunet_cls":
        return NNUNetClassifier(in_channels=in_channels, num_classes=num_classes)

    if model_name == "swin_unetr_cls":
        return SwinUNETRClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            feature_size=24,
        )

    if model_name == "efficientnetb0_unet_cls":
        return EfficientNetB0Classifier(
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=True,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")