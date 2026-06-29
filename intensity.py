class RawRescale:
    
    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            out = np.zeros_like(arr, dtype=np.uint8)
        else:
            out = (arr - mn) / (mx - mn)
            out = (out * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out)


class PercentileClip:
    
    def __init__(self, lower=0.5, upper=99.5):
        self.lower = lower
        self.upper = upper

    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        lo = np.percentile(arr, self.lower)
        hi = np.percentile(arr, self.upper)
        arr = np.clip(arr, lo, hi)

        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            out = np.zeros_like(arr, dtype=np.uint8)
        else:
            out = (arr - mn) / (mx - mn)
            out = (out * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out)


class HistogramMatchTransform:
    
    def __init__(self, ref_img_array):
        self.ref = ref_img_array.astype(np.float32)

    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        matched = exposure.match_histograms(arr, self.ref, channel_axis=None)
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        return Image.fromarray(matched)


class CLAHETransform:
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        arr = np.array(img).astype(np.uint8)
        out = self.clahe.apply(arr)
        return Image.fromarray(out)


def zscore_per_image_tensor(tensor):
    
    mean = tensor.mean()
    std = tensor.std()
    std = std if std > 1e-8 else tensor.new_tensor(1.0)
    return (tensor - mean) / std


def zscore_per_dataset_tensor(tensor, mean, std):
    
    std = std if std > 1e-8 else 1.0
    return (tensor - mean) / std


def get_intensity_pre_pil(mode, ref_img_array=None):
    
    if mode == "raw_rescale":
        return [RawRescale()]
    if mode == "zscore_per_image":
        return []
    if mode == "zscore_per_dataset":
        return []
    if mode == "percentile_zscore":
        return [PercentileClip(0.5, 99.5)]
    if mode == "hist_match":
        if ref_img_array is None:
            raise ValueError("hist_match requires ref_img_array")
        return [HistogramMatchTransform(ref_img_array)]
    if mode == "clahe_zscore":
        return [CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))]
    raise ValueError(f"Unknown intensity mode: {mode}")