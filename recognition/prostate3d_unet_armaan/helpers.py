import numpy as np

def to_channels(label_volume, num_classes=None, dtype=np.uint8):
    if num_classes is None:
        num_classes = int(label_volume.max() + 1)
    shape = label_volume.shape + (num_classes,)
    out = np.zeros(shape, dtype=dtype)
    for c in range(num_classes):
        out[..., c] = (label_volume == c).astype(dtype)
    return out

def applyOrientation(niftiImage, interpolation='linear', scale=1):
    return niftiImage