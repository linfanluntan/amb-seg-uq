from .losses import DiceCELoss, EvidentialLoss, DistributionalLoss


def build_loss(config):
    """Factory function to build loss from config."""
    loss_name = config["training"]["loss"]["name"]
    
    if loss_name == "dice_ce":
        return DiceCELoss(
            dice_weight=config["training"]["loss"].get("dice_weight", 1.0),
            ce_weight=config["training"]["loss"].get("ce_weight", 1.0),
        )
    elif loss_name == "evidential":
        return EvidentialLoss(
            kl_weight=config["training"]["loss"].get("kl_weight", 0.05),
            annealing_epochs=config["model"].get("evidential", {}).get("annealing_epochs", 10),
            dice_weight=config["training"]["loss"].get("dice_weight", 0.5),
        )
    elif loss_name == "distributional":
        return DistributionalLoss(
            divergence=config["model"].get("multi_annotator", {}).get("loss", "kl"),
            dice_weight=config["training"]["loss"].get("dice_weight", 0.5),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
