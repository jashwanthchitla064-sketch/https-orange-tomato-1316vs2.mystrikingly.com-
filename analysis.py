"""FieldHealth core: vegetation index -> mask -> block grid -> classification."""
import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None
DEFAULTS = {"green": 0.6, "yellow": 0.3, "block": 20, "index": "VARI"}
COLORS = {0: (46, 204, 64), 1: (255, 220, 0), 2: (231, 76, 60)}  # green/yellow/red


class AnalysisError(Exception):
    """User-facing error (bad file, too dark, no vegetation...)."""


def load(path):
    """Return (rgb float32 HxWx3, nir float32 HxW or None). TIFFs with >=4 bands = R,G,B,NIR."""
    try:
        if str(path).lower().endswith((".tif", ".tiff")):
            try:
                import rasterio
                with rasterio.open(path) as ds:
                    arr = ds.read().astype(np.float32)
                if arr.shape[0] >= 3:
                    arr = arr / max(float(arr.max()), 1.0) * 255
                    rgb = np.moveaxis(arr[:3], 0, -1)
                    return rgb, (arr[3] if arr.shape[0] >= 4 else None)
            except ImportError:
                pass
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32), None
    except Exception as e:
        raise AnalysisError(f"Unsupported or corrupt image: {e}")


def vegetation_index(rgb, nir=None, method="VARI"):
    """NDVI if NIR given; else VARI=(G-R)/(G+R-B) or ExG=2g-r-b (chromatic coords). Result in [-1,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if nir is not None:
        den = nir + r
        out = np.where(den > 1e-3, (nir - r) / np.maximum(den, 1e-3), 0)
    elif method == "ExG":
        s = r + g + b + 1e-3
        out = (2 * g - r - b) / s
    else:
        den = g + r - b
        out = np.where(np.abs(den) > 1e-3, (g - r) / np.where(np.abs(den) > 1e-3, den, 1), 0)
    return np.clip(out, -1, 1).astype(np.float32)


def crop_mask(rgb):
    """True = candidate crop pixel. Drops shadows, water (blue-dominant) and gray (roads/buildings).
    Bare soil is KEPT on purpose: bare patches inside a field are problems (score Red)."""
    mx, mn = rgb.max(2), rgb.min(2)
    sat = (mx - mn) / (mx + 1e-3)
    water = (rgb[..., 2] > rgb[..., 1]) & (rgb[..., 2] > rgb[..., 0])
    return ~((mx < 40) | (sat < 0.1) | water)


def check_quality(rgb):
    gray = rgb.mean(2)[::2, ::2]
    if gray.mean() < 30:
        raise AnalysisError("Image is too dark to analyze.")
    lap = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return ["Image looks blurry; results may be unreliable."] if lap.var() < 15 else []


def block_stats(index, mask, bs, min_crop=0.3):
    """Mean index per bs x bs block over crop pixels. Returns (means, valid) arrays."""
    H, W = (index.shape[0] // bs) * bs, (index.shape[1] // bs) * bs
    if H == 0 or W == 0:
        raise AnalysisError("Image smaller than one grid block.")
    sh = (H // bs, bs, W // bs, bs)
    m = mask[:H, :W].reshape(sh).astype(np.float32)
    v = (index[:H, :W].reshape(sh)) * m
    cnt = m.sum((1, 3))
    means = v.sum((1, 3)) / np.maximum(cnt, 1)
    return means, cnt / (bs * bs) >= min_crop


def classify(means, valid, green=0.6, yellow=0.3):
    """Labels: 0 green (>=green), 1 yellow (yellow<=i<green), 2 red (<yellow), -1 excluded."""
    if yellow > green:
        raise ValueError("yellow threshold must be <= green threshold")
    lab = np.full(means.shape, -1, dtype=np.int8)
    lab[valid & (means >= green)] = 0
    lab[valid & (means >= yellow) & (means < green)] = 1
    lab[valid & (means < yellow)] = 2
    return lab


def percentages(labels):
    """[green%, yellow%, red%] over classified blocks; sums to 100."""
    c = np.array([(labels == k).sum() for k in (0, 1, 2)], dtype=float)
    if c.sum() == 0:
        raise AnalysisError("No vegetation detected in this image.")
    return list(c / c.sum() * 100)


def problem_zones(labels, bs, top=5):
    """Largest connected red clusters, with bbox (px) and size in blocks."""
    lab, n = ndimage.label(labels == 2)
    zones = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        blocks = int((lab[sl] == i).sum())
        zones.append({"blocks": blocks, "x": sl[1].start * bs, "y": sl[0].start * bs,
                      "w": (sl[1].stop - sl[1].start) * bs, "h": (sl[0].stop - sl[0].start) * bs})
    return sorted(zones, key=lambda z: -z["blocks"])[:top]


def render_overlay(rgb, labels, bs, alpha=0.45):
    out = rgb.copy()
    H, W = labels.shape[0] * bs, labels.shape[1] * bs
    for k, col in COLORS.items():
        m = np.kron(labels == k, np.ones((bs, bs), dtype=bool))
        reg = out[:H, :W]
        reg[m] = reg[m] * (1 - alpha) + np.array(col, dtype=np.float32) * alpha
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def analyze_image(path, s):
    rgb, nir = load(path)
    warnings = check_quality(rgb)
    idx = vegetation_index(rgb, nir, s["index"])
    means, valid = block_stats(idx, crop_mask(rgb), int(s["block"]))
    labels = classify(means, valid, s["green"], s["yellow"])
    pct = percentages(labels)
    return rgb, labels, means, pct, warnings
