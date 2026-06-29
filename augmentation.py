def get_augmentation_transform(mode, output_size):
    
    if mode == "none":
        return []

    if mode == "minimal":
        return [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=5),
        ]

    if mode == "moderate":
        return [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(size=output_size, scale=(0.8, 1.0)),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]

    if mode == "aggressive":
        return [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(size=output_size, scale=(0.7, 1.0)),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        ]

    raise ValueError(f"Unknown augmentation mode: {mode}")