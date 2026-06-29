def get_resolution_transform(mode):
    
    if mode == "native":
        return []
    if mode in ["128", "224", "384", "512", "768"]:
        size = int(mode)
        return [T.Resize((size, size))]
    raise ValueError(f"Unknown resolution mode: {mode}")