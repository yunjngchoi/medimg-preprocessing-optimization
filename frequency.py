def _to_uint8(arr):
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


class WaveletHFChannels:
    
    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        _, (lh, hl, hh) = pywt.dwt2(arr, "haar")
        h, w = arr.shape
        lh = cv2.resize(lh, (w, h))
        hl = cv2.resize(hl, (w, h))
        hh = cv2.resize(hh, (w, h))
        return np.stack([_to_uint8(arr), _to_uint8(lh), _to_uint8(hl), _to_uint8(hh)], axis=0)


class LoGEdgeChannel:
    
    def __init__(self, sigma=1.5):
        self.sigma = sigma

    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        edge = ndimage.gaussian_laplace(arr, sigma=self.sigma)
        return np.stack([_to_uint8(arr), _to_uint8(edge)], axis=0)


class FFTHighPassChannel:
    
    def __init__(self, keep_ratio=0.15):
        self.keep_ratio = keep_ratio

    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)

        h, w = arr.shape
        cy, cx = h // 2, w // 2
        ry, rx = int(h * self.keep_ratio / 2), int(w * self.keep_ratio / 2)

        mask = np.ones((h, w), dtype=np.float32)
        mask[cy - ry:cy + ry, cx - rx:cx + rx] = 0.0

        hf = fshift * mask
        rec = np.fft.ifft2(np.fft.ifftshift(hf))
        rec = np.abs(rec)

        return np.stack([_to_uint8(arr), _to_uint8(rec)], axis=0)


class GaborBankChannels:
    
    def __init__(self, thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), lambd=6.0):
        self.thetas = thetas
        self.lambd = lambd

    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        channels = [_to_uint8(arr * 255.0)]
        for theta in self.thetas:
            real, _ = gabor(arr, frequency=1.0 / self.lambd, theta=theta)
            channels.append(_to_uint8(real))
        return np.stack(channels, axis=0)


class MultiScaleLoGChannels:
    
    def __init__(self, sigmas=(1.0, 2.5, 5.0)):
        self.sigmas = sigmas

    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        channels = [_to_uint8(arr)]
        for sigma in self.sigmas:
            edge = ndimage.gaussian_laplace(arr, sigma=sigma)
            channels.append(_to_uint8(edge))
        return np.stack(channels, axis=0)


def get_frequency_transform(mode):
    
    if mode == "none":
        return None
    if mode == "wavelet_hf":
        return WaveletHFChannels()
    if mode == "log_edge":
        return LoGEdgeChannel(sigma=1.5)
    if mode == "fft_highpass":
        return FFTHighPassChannel(keep_ratio=0.15)
    if mode == "gabor_bank":
        return GaborBankChannels()
    if mode == "multiscale_log":
        return MultiScaleLoGChannels()
    raise ValueError(f"Unknown frequency mode: {mode}")