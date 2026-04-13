# Models package
from .unet import UNet
from .resenc_unet3d import ResEncUNet3D
from .mc_dropout import MCDropoutWrapper
from .ensemble import DeepEnsemble
from .evidential_head import EvidentialHead, evidential_uncertainty_decomposition


def build_model(config):
    """Factory function to build model from config."""
    arch = config["model"].get("architecture", "unet2d")
    evidential = config["model"].get("evidential", {}).get("enabled", False)
    dropout = config["model"].get("dropout_rate", config["model"].get("dropout", 0.0))
    
    # MC Dropout uses dropout in model
    if config.get("uncertainty", {}).get("mc_dropout", {}).get("enabled", False):
        dropout = config["uncertainty"]["mc_dropout"].get("dropout_rate", 0.1)
    
    if arch in ("unet2d", "unet"):
        model = UNet(
            in_channels=config["model"]["in_channels"],
            num_classes=config["model"]["num_classes"],
            features=config["model"].get("features", [32, 64, 128, 256, 512]),
            dropout_rate=dropout,
            norm=config["model"].get("norm", "instance"),
            activation=config["model"].get("activation", "leakyrelu"),
            dim=2,
            evidential=evidential,
            evidential_activation=config["model"].get("evidential", {}).get("activation", "softplus"),
        )
    elif arch == "unet3d":
        model = UNet(
            in_channels=config["model"]["in_channels"],
            num_classes=config["model"]["num_classes"],
            features=config["model"].get("features", [32, 64, 128, 256]),
            dropout_rate=dropout,
            norm=config["model"].get("norm", "instance"),
            dim=3,
            evidential=evidential,
        )
    elif arch in ("resenc3d", "resenc_unet3d"):
        patch_size = tuple(config.get("preprocessing", {}).get("patch_size", [64, 128, 128]))
        model = ResEncUNet3D(
            in_channels=config["model"]["in_channels"],
            num_classes=config["model"]["num_classes"],
            patch_size=patch_size,
            base_features=config["model"].get("base_features", 32),
            max_features=config["model"].get("max_features", 320),
            n_stages=config["model"].get("n_stages", 5),
            blocks_per_stage=config["model"].get("blocks_per_stage", 2),
            dropout=dropout,
            deep_supervision=config["model"].get("deep_supervision", True),
            evidential=evidential,
            evidential_activation=config["model"].get("evidential", {}).get("activation", "softplus"),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Wrap for MC Dropout
    if config.get("uncertainty", {}).get("mc_dropout", {}).get("enabled", False):
        model = MCDropoutWrapper(
            model,
            num_samples=config["uncertainty"]["mc_dropout"].get("num_samples", 20),
            dropout_rate=dropout,
        )
    
    return model
