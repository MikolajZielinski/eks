import sys
import math
import argparse
import numpy as np
from PIL import Image
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def get_model_and_timestamp():
    parser = argparse.ArgumentParser(description="Scale and center a mesh")
    parser.add_argument("--model-type", default='chair', dest="model_type")
    parser.add_argument("--timestamp", default='new_denisfy', dest="timestamp")
    args = parser.parse_args()

    return args.model_type, args.timestamp

def load_image(path: Path):
    try:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im)
        im.save("temp_debug.png")  # debug
        return arr
    except Exception as e:
        print(f"Warning: failed to load '{path}': {e}", file=sys.stderr)
        return None


def infer_data_range(arr: np.ndarray) -> float:
    if np.issubdtype(arr.dtype, np.floating):
        # assume floats in 0..1 if max <= 1.0
        return 1.0 if arr.max() <= 1.0 else float(arr.max())
    if np.issubdtype(arr.dtype, np.integer):
        # common integer ranges
        info = np.iinfo(arr.dtype)
        return float(info.max)
    return float(arr.max())


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = None) -> float:
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    mse = np.mean((a_f - b_f) ** 2)
    if mse == 0:
        return float("inf")
    if data_range is None:
        # choose max of inferred ranges to be safe
        data_range = max(infer_data_range(a), infer_data_range(b))
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def gather_images(folder: Path):
    files = {}
    if not folder.exists():
        print(f"Error: folder '{folder}' does not exist", file=sys.stderr)
        return files
    iterator = folder.glob("*")
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        key = p.stem  # filename without extension
        # if multiple files have same stem, keep the first found (could be extended)
        if key not in files:
            files[key] = p
    return files


def main():

    model_type, timestamp = get_model_and_timestamp()

    default_a = Path(f"renders/{model_type}/{timestamp}/test/rgb")
    default_b = Path(f"blender_neuraleditor/{model_type}/views_test")

    a_files = gather_images(default_a)
    b_files = gather_images(default_b)

    keys_a = sorted(a_files.keys())
    keys_b = sorted(b_files.keys())

    psnr_values = []
    print(f"Found {len(a_files)} matching files. Computing PSNR...")
    for key in range(len(a_files)):
        pa = a_files[keys_a[key]]
        pb = b_files[keys_b[key]]
        a_img = load_image(pa)
        b_img = load_image(pb)
        if a_img is None or b_img is None:
            print(f"Skipping '{key}': failed to load image.", file=sys.stderr)
            continue
        if a_img.shape != b_img.shape:
            print(f"Skipping '{key}': size mismatch {a_img.shape} vs {b_img.shape}", file=sys.stderr)
            continue
        value = psnr(a_img, b_img, data_range=None)
        psnr_values.append(value)
        valstr = "inf" if math.isinf(value) else f"{value:.3f}"
        print(f"{key}: PSNR = {valstr} dB")

    if not psnr_values:
        print("No PSNR values computed.", file=sys.stderr)
        return 1

    finite_vals = [v for v in psnr_values if not math.isinf(v)]
    mean_psnr = float("inf") if len(finite_vals) == 0 and any(math.isinf(v) for v in psnr_values) else (sum(finite_vals) / len(finite_vals) if finite_vals else 0.0)
    count_inf = sum(1 for v in psnr_values if math.isinf(v))
    print("----")
    print(f"Computed {len(psnr_values)} PSNRs ({count_inf} infinite).")
    if math.isinf(mean_psnr):
        print("Average PSNR (finite-only): all infinite (perfect matches).")
    else:
        print(f"Average PSNR (finite-only) = {mean_psnr:.3f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())