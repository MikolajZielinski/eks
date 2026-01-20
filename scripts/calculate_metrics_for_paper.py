import sys
import math
import torch
import lpips
import argparse
import numpy as np
import torch.nn.functional as F

from math import exp
from PIL import Image
from pathlib import Path
from torch.autograd import Variable


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_model_and_timestamp():
    parser = argparse.ArgumentParser(description="Scale and center a mesh")
    parser.add_argument("--model-type", default='chair', dest="model_type")
    parser.add_argument("--timestamp", default='final_black', dest="timestamp")
    args = parser.parse_args()

    return args.model_type, args.timestamp

def load_image(path: Path):
    try:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im)
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

    model_type_arg, timestamp = get_model_and_timestamp()

    scenes = ['chair', 'drums', 'lego', 'mic', 'materials', 'ship', 'hotdog', 'ficus']

    # Initialize LPIPS
    loss_fn = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    results = {}

    for model_type in scenes:
        print(f"Processing scene: {model_type}")
        default_a = Path(f"renders/neuraleditor/{model_type}/{timestamp}/test/rgb")
        default_b = Path(f"renders/neuraleditor/{model_type}/{timestamp}/target")
        error_dir = default_a.parent / "error"
        error_dir.mkdir(parents=True, exist_ok=True)

        if not default_a.exists():
            print(f"  Folder not found: {default_a}")
            continue

        scene_psnr = []
        scene_ssim = []
        scene_lpips = []
        a_files = gather_images(default_a)
        b_files = gather_images(default_b)

        keys_a = sorted(a_files.keys())
        keys_b = sorted(b_files.keys())

        for key in range(len(a_files)):
            pa = a_files[keys_a[key]]
            pb = b_files[keys_b[key]]
            a_img_np = load_image(pa)
            b_img_np = load_image(pb)

            if a_img_np is None or b_img_np is None:
                continue
            if a_img_np.shape != b_img_np.shape:
                continue
            
            # Save error map
            a_f = a_img_np.astype(np.float32) / 255.0
            b_f = b_img_np.astype(np.float32) / 255.0
            error = np.clip(np.mean(np.abs(a_f - b_f), axis=-1), 0.0, 1.0)
            err_img = Image.fromarray((error * 255).astype(np.uint8), mode="L")
            err_img.save(error_dir / f"{pa.stem}.png")

            # PSNR
            p = psnr(a_img_np, b_img_np, data_range=None)
            scene_psnr.append(p)
            
            # Prepare tensors for SSIM & LPIPS
            # Normalize to [0, 1]
            t_a = torch.from_numpy(a_img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            t_b = torch.from_numpy(b_img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            if torch.cuda.is_available():
                t_a = t_a.cuda()
                t_b = t_b.cuda()
                
            # SSIM
            s = ssim(t_a, t_b, window_size=11, size_average=True).item()
            scene_ssim.append(s)
            
            # LPIPS (requires [-1, 1])
            lp = loss_fn(t_a * 2.0 - 1.0, t_b * 2.0 - 1.0).item()
            scene_lpips.append(lp)

            # print(f"Image {pa.name}: PSNR={p:.2f}, SSIM={s:.4f}, LPIPS={lp:.4f}")

        mean_psnr = np.mean(scene_psnr) if scene_psnr else float('nan')
        mean_ssim = np.mean(scene_ssim) if scene_ssim else float('nan')
        mean_lpips = np.mean(scene_lpips) if scene_lpips else float('nan')
        
        results[model_type] = {
            'psnr': mean_psnr,
            'ssim': mean_ssim,
            'lpips': mean_lpips
        }

        # Save metric file in default_a (rgb folder)
        metric_path = default_a / "metrics.txt"
        with open(metric_path, "w") as f:
            f.write(f"PSNR: {mean_psnr}\n")
            f.write(f"SSIM: {mean_ssim}\n")
            f.write(f"LPIPS: {mean_lpips}\n")
        print(f"  Saved metrics to {metric_path}")

    # Print LaTeX ready form
    print("\nLaTeX Metrics:")
    print(" & ".join(scenes))
    
    row_psnr = []
    row_ssim = []
    row_lpips = []
    
    for s in scenes:
        if s in results:
            # Match format example: 25.85
            row_psnr.append(f"{results[s]['psnr']:.2f}")
            row_ssim.append(f"{results[s]['ssim']:.3f}")
            row_lpips.append(f"{results[s]['lpips']:.3f}")
        else:
            row_psnr.append("-")
            row_ssim.append("-")
            row_lpips.append("-")

    print(" & ".join(row_psnr))
    print(" & ".join(row_ssim))
    print(" & ".join(row_lpips))



if __name__ == "__main__":
    sys.exit(main())