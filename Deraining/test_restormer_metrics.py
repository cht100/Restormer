## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import argparse
import csv
import json
import logging
import math
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from glob import glob
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import yaml
from basicsr.models.archs.restormer_arch import Restormer
from natsort import natsorted
from PIL import Image
from skimage import img_as_ubyte
from tqdm import tqdm

try:
    import pyiqa
except ImportError as exc:
    raise ImportError(
        "需要安装 pyiqa 才能计算 LPIPS/PSNR/SSIM/MANIQA/CLIPIQA/MUSIQ/NIQE：pip install pyiqa"
    ) from exc

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    HAS_FID = True
    FID_IMPORT_ERROR = None
except ImportError as exc:
    FrechetInceptionDistance = None
    HAS_FID = False
    FID_IMPORT_ERROR = exc


# ======================================================================================
# Eval-style constants / utilities
# ======================================================================================

FIXED_RESULTS_ROOT = "/home/quyu/16-baselines-restoration/results"
MODEL_TYPE = "restormer"

LOWER_BETTER = {"LPIPS", "NIQE", "FID"}
HIGHER_BETTER = {"PSNR", "SSIM", "MANIQA", "CLIPIQA", "MUSIQ"}
PER_OUTPUT_METRICS = ["LPIPS", "PSNR", "SSIM", "MANIQA", "CLIPIQA", "MUSIQ", "NIQE"]
SUMMARY_METRICS = PER_OUTPUT_METRICS + ["FID"]
OUTPUT_KEYS = ["final"]


def direction_symbol(name: str) -> str:
    return "↓" if name in LOWER_BETTER else "↑"


def safe_dataset_filename(name: str) -> str:
    name = str(name).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    for ch in [":", "*", "?", '"', "<", ">", "|"]:
        name = name.replace(ch, "_")
    return name or "dataset"


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def force_write_text(path: str, text: str) -> None:
    mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def force_write_json(path: str, obj: dict) -> None:
    mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())


def force_write_csv(path: str, rows: List[List[object]]) -> None:
    mkdir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def build_run_save_root(base_results_root: str, model_type: str, dataset_name: str, run_time: str) -> str:
    """
    和 eval_unified.py 一致的目录：
        {base_results_root}/{model_type}/{dataset_name}/{run_time}
    默认：
        /home/quyu/16-baselines-restoration/results/restormer/{dataset}/{run_time}
    """
    return os.path.join(
        base_results_root,
        safe_dataset_filename(model_type),
        safe_dataset_filename(dataset_name),
        run_time,
    )


# ======================================================================================
# Result files: TXT / CSV / JSON
# ======================================================================================


def get_result_paths(results_root: str, dataset_name: str) -> Tuple[str, str, str, str, str, str]:
    """
    results_root 已经是完整运行目录：
        /home/quyu/16-baselines-restoration/results/restormer/{dataset}/{run_time}
    """
    dataset_dir = results_root
    output_dir = os.path.join(dataset_dir, "output")
    combined_dir = os.path.join(dataset_dir, "combined")
    file_stem = f"metrics_results_{safe_dataset_filename(dataset_name)}"
    txt_path = os.path.join(dataset_dir, file_stem + ".txt")
    csv_path = os.path.join(dataset_dir, file_stem + ".csv")
    json_path = os.path.join(dataset_dir, file_stem + ".json")
    return dataset_dir, output_dir, combined_dir, txt_path, csv_path, json_path


def create_init_result_files(
    results_root: str,
    dataset_name: str,
    total: int,
    output_keys: Iterable[str],
) -> Tuple[str, str, str, str, str, str]:
    dataset_dir, output_dir, combined_dir, txt_path, csv_path, json_path = get_result_paths(results_root, dataset_name)
    mkdir(dataset_dir)
    mkdir(output_dir)
    mkdir(combined_dir)

    output_key_list = list(output_keys)
    init_text = (
        "Image Quality Metrics Evaluation\n"
        "Status: INIT\n"
        f"Dataset: {dataset_name}\n"
        f"Processed/Total: 0/{total}\n"
        "Data/model pipeline: official Restormer test.py pipeline\n"
        "Evaluation style: average over dataset for each output key\n"
        "Metrics: LPIPS / PSNR / SSIM / MANIQA / CLIPIQA / MUSIQ / NIQE / FID\n"
        f"output_keys: {output_key_list}\n"
        "Output pictures: output/\n"
        "Combined pictures: combined/Input_GT_output\n"
    )
    force_write_text(txt_path, init_text)
    force_write_csv(csv_path, [["status", "dataset", "processed", "total", "output_key", "valid_outputs"] + SUMMARY_METRICS])
    force_write_json(json_path, {
        "status": "INIT",
        "dataset": dataset_name,
        "processed": 0,
        "total": total,
        "output_keys": output_key_list,
        "metrics": SUMMARY_METRICS,
        "average_by_output": {},
        "valid_outputs_by_output": {},
    })

    print("\n[结果文件已创建]")
    print(f"TXT : {txt_path}")
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"OUTPUT_DIR  : {output_dir}")
    print(f"COMBINED_DIR: {combined_dir}\n")
    return dataset_dir, output_dir, combined_dir, txt_path, csv_path, json_path


# ======================================================================================
# Image conversion / visualization
# ======================================================================================


def ensure_3ch_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return arr


def rgb_uint8_to_bgr_uint8(img_rgb: np.ndarray) -> np.ndarray:
    arr = ensure_3ch_uint8(img_rgb)
    return arr[:, :, ::-1].copy()


def bgr_uint8_to_rgb_tensor01(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """内部统一约定：指标计算用 RGB [0,1] NCHW；这里从 BGR uint8 转过去。"""
    arr = ensure_3ch_uint8(img)[:, :, ::-1].copy()
    return torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


def bgr_uint8_to_pil_rgb(img: np.ndarray) -> Image.Image:
    arr = ensure_3ch_uint8(img)[:, :, ::-1].copy()
    return Image.fromarray(arr, mode="RGB")


def crop_pair(sr_img: np.ndarray, gt_img: np.ndarray, crop_border: int) -> Tuple[np.ndarray, np.ndarray]:
    if crop_border <= 0:
        return sr_img, gt_img
    h, w = sr_img.shape[:2]
    if h <= crop_border * 2 or w <= crop_border * 2:
        return sr_img, gt_img
    return sr_img[crop_border:-crop_border, crop_border:-crop_border], gt_img[crop_border:-crop_border, crop_border:-crop_border]


def save_combined(
    lq_img: Optional[np.ndarray],
    gt_img: Optional[np.ndarray],
    outputs: "OrderedDict[str, np.ndarray]",
    save_path: str,
) -> None:
    tiles: List[Image.Image] = []
    if lq_img is not None:
        tiles.append(bgr_uint8_to_pil_rgb(lq_img))
    if gt_img is not None:
        tiles.append(bgr_uint8_to_pil_rgb(gt_img))
    for img in outputs.values():
        tiles.append(bgr_uint8_to_pil_rgb(img))
    if not tiles:
        return

    base_size = tiles[0].size
    tiles = [tile if tile.size == base_size else tile.resize(base_size, Image.LANCZOS) for tile in tiles]
    w, h = base_size
    canvas = Image.new("RGB", (w * len(tiles), h))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * w, 0))
    mkdir(os.path.dirname(save_path))
    canvas.save(save_path)


# ======================================================================================
# Metric accumulation / writing
# ======================================================================================


def init_metric_storage(output_keys: Iterable[str]) -> Tuple[OrderedDict, OrderedDict, OrderedDict]:
    sums = OrderedDict()
    counts = OrderedDict()
    output_counts = OrderedDict()
    for key in output_keys:
        sums[key] = defaultdict(float)
        counts[key] = defaultdict(int)
        output_counts[key] = 0
    return sums, counts, output_counts


def ensure_output_key(key: str, sums: OrderedDict, counts: OrderedDict, output_counts: OrderedDict) -> None:
    if key not in sums:
        sums[key] = defaultdict(float)
        counts[key] = defaultdict(int)
        output_counts[key] = 0


def add_metric(sums: OrderedDict, counts: OrderedDict, key: str, metric: str, value: float) -> None:
    try:
        value = float(value)
    except Exception:
        return
    if math.isnan(value):
        return
    sums[key][metric] += value
    counts[key][metric] += 1


def compute_averages(
    sums: OrderedDict,
    counts: OrderedDict,
    fid_values: Optional[Dict[str, float]] = None,
) -> "OrderedDict[str, Dict[str, float]]":
    averages = OrderedDict()
    for key in sums.keys():
        averages[key] = {}
        for metric in PER_OUTPUT_METRICS:
            n = counts[key].get(metric, 0)
            averages[key][metric] = sums[key][metric] / n if n > 0 else float("nan")
        averages[key]["FID"] = fid_values.get(key, float("nan")) if fid_values is not None else float("nan")
    return averages


def compute_one_output_metrics(
    sr_img: np.ndarray,
    gt_img: Optional[np.ndarray],
    need_gt: bool,
    crop_border: int,
    fr_metrics: Dict[str, object],
    nr_metrics: Dict[str, object],
    device: torch.device,
    logger: logging.Logger,
    img_name: str,
    output_key: str,
) -> Dict[str, float]:
    result = {m: float("nan") for m in PER_OUTPUT_METRICS}

    sr_rgb = bgr_uint8_to_rgb_tensor01(sr_img, device)

    if need_gt and gt_img is not None:
        cropped_sr, cropped_gt = crop_pair(sr_img, gt_img, crop_border)
        if cropped_sr.shape == cropped_gt.shape:
            sr_crop_rgb = bgr_uint8_to_rgb_tensor01(cropped_sr, device)
            gt_crop_rgb = bgr_uint8_to_rgb_tensor01(cropped_gt, device)
            for metric in ("LPIPS", "PSNR", "SSIM"):
                try:
                    result[metric] = float(fr_metrics[metric](sr_crop_rgb, gt_crop_rgb).item())
                except Exception as exc:
                    logger.warning("%s failed for %s/%s: %s", metric, img_name, output_key, exc)
        else:
            logger.warning(
                "Skip FR metrics for %s/%s because SR shape %s != GT shape %s",
                img_name, output_key, cropped_sr.shape, cropped_gt.shape,
            )

    for metric, fn in nr_metrics.items():
        try:
            result[metric] = float(fn(sr_rgb).item())
        except Exception as exc:
            logger.warning("%s failed for %s/%s: %s", metric, img_name, output_key, exc)

    return result


def is_better(metric: str, new_value: float, old_value: float) -> bool:
    if math.isnan(new_value):
        return False
    if math.isnan(old_value):
        return True
    return new_value < old_value if metric in LOWER_BETTER else new_value > old_value


def best_output(averages: "OrderedDict[str, Dict[str, float]]", metric: str) -> str:
    best_key = "N/A"
    best_value = float("nan")
    for key, values in averages.items():
        v = values.get(metric, float("nan"))
        if best_key == "N/A" or is_better(metric, v, best_value):
            best_key, best_value = key, v
    return best_key


def write_results(
    txt_path: str,
    csv_path: str,
    json_path: str,
    dataset_name: str,
    processed: int,
    total: int,
    output_keys: Iterable[str],
    averages: "OrderedDict[str, Dict[str, float]]",
    metric_counts: OrderedDict,
    output_counts: OrderedDict,
    avg_time: float,
    final: bool,
    fid_counts: Optional[Dict[str, int]] = None,
) -> None:
    status = "FINAL" if final else "RUNNING"
    output_key_list = list(output_keys)
    fid_counts = fid_counts or {}

    lines = []
    lines.append("Image Quality Metrics Evaluation")
    lines.append(f"Status: {status}")
    lines.append(f"Dataset: {dataset_name}")
    lines.append(f"Processed/Total: {processed}/{total}")
    lines.append("Data/model pipeline: official Restormer test.py pipeline")
    lines.append("Evaluation style: average over dataset for each output key")
    lines.append("Metrics: LPIPS / PSNR / SSIM / MANIQA / CLIPIQA / MUSIQ / NIQE / FID")
    lines.append(f"output_keys: {output_key_list}")
    lines.append("Output pictures: output/")
    lines.append("Combined pictures: combined/Input_GT_output")
    lines.append(f"Average test time: {avg_time:.6f} sec/sample" if not math.isnan(avg_time) else "Average test time: nan")
    lines.append("=" * 120)
    lines.append("")

    for key, values in averages.items():
        lines.append(f"************************** OUTPUT_{key} **************************")
        lines.append(f"valid_outputs: {output_counts.get(key, 0)}")
        lines.append(f"{'Metric':<16} {'Average':>14} {'Direction':>10} {'Valid':>10}")
        lines.append("-" * 80)
        for metric in SUMMARY_METRICS:
            value = values.get(metric, float("nan"))
            value_str = "nan" if math.isnan(value) else f"{value:.6f}"
            valid = fid_counts.get(key, 0) if metric == "FID" else metric_counts.get(key, {}).get(metric, 0)
            lines.append(
                f"{metric + '(' + direction_symbol(metric) + ')':<16} "
                f"{value_str:>14} {'better=' + direction_symbol(metric):>10} {valid:>10}"
            )
        lines.append("")

    if final:
        lines.append("Best output by metric")
        lines.append("-" * 80)
        for metric in SUMMARY_METRICS:
            key = best_output(averages, metric)
            value = averages.get(key, {}).get(metric, float("nan")) if key != "N/A" else float("nan")
            value_str = "nan" if math.isnan(value) else f"{value:.6f}"
            lines.append(f"{metric:<16} best={key:<12} value={value_str}")

    force_write_text(txt_path, "\n".join(lines) + "\n")

    csv_rows = [["status", "dataset", "processed", "total", "output_key", "valid_outputs"] + SUMMARY_METRICS]
    for key, values in averages.items():
        row = [status, dataset_name, processed, total, key, output_counts.get(key, 0)]
        for metric in SUMMARY_METRICS:
            value = values.get(metric, float("nan"))
            row.append("nan" if math.isnan(value) else f"{value:.6f}")
        csv_rows.append(row)
    force_write_csv(csv_path, csv_rows)

    json_obj = {
        "status": status,
        "dataset": dataset_name,
        "processed": processed,
        "total": total,
        "average_test_time": None if math.isnan(avg_time) else avg_time,
        "output_keys": output_key_list,
        "metrics": SUMMARY_METRICS,
        "average_by_output": {
            key: {metric: (None if math.isnan(value) else round(value, 6)) for metric, value in values.items()}
            for key, values in averages.items()
        },
        "valid_outputs_by_output": dict(output_counts),
        "valid_fid_by_output": dict(fid_counts),
    }
    if final:
        json_obj["best_output_by_metric"] = {metric: best_output(averages, metric) for metric in SUMMARY_METRICS}
    force_write_json(json_path, json_obj)


# ======================================================================================
# Restormer official test helpers
# ======================================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image Deraining using Restormer")
    parser.add_argument("--input_dir", default="/home/quyu/datasets/Rain13K/", type=str, help="Directory of validation images")
    parser.add_argument(
        "--result_dir",
        default=FIXED_RESULTS_ROOT,
        type=str,
        help="结果保存根目录；最终会自动拼成 result_dir/restormer/dataset/time/",
    )
    parser.add_argument("--weights", default="/home/quyu/16-baselines-restoration/models/Restormer/Derain/net_g_300000.pth", type=str, help="Path to weights")
    parser.add_argument("--limit", type=int, default=None, help="只测试每个数据集前 N 张；默认测试全部。")
    parser.add_argument("--no_fid", action="store_true", help="跳过 FID。")
    parser.add_argument("--write_every", type=int, default=2, help="每处理 N 张更新一次 TXT/CSV/JSON；默认 5。")
    parser.add_argument("--crop_border", type=int, default=0, help="计算 PSNR/SSIM/LPIPS 前裁边；默认 0。")
    return parser.parse_args()


def find_gt_path(input_file: str, target_dir: Optional[str]) -> Optional[str]:
    if target_dir is None or not os.path.isdir(target_dir):
        return None

    basename = os.path.basename(input_file)
    same_name = os.path.join(target_dir, basename)
    if os.path.isfile(same_name):
        return same_name

    stem = os.path.splitext(basename)[0]
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = os.path.join(target_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def find_target_dir(input_dir: str, dataset: str) -> Optional[str]:
    # Restormer 官方 deraining 数据通常是 test/{dataset}/input 和 test/{dataset}/target
    base = os.path.dirname(input_dir)
    for name in ("target", "gt", "GT", "groundtruth", "ground_truth", "label", "clean"):
        candidate = os.path.join(base, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def init_metrics(device: torch.device):
    print(f"[指标初始化] device={device}")
    fr_metrics = {
        "LPIPS": pyiqa.create_metric("lpips", device=device),
        "PSNR": pyiqa.create_metric("psnr", device=device),
        "SSIM": pyiqa.create_metric("ssim", device=device),
    }
    nr_metrics = {
        "MANIQA": pyiqa.create_metric("maniqa", device=device),
        "CLIPIQA": pyiqa.create_metric("clipiqa", device=device),
        "MUSIQ": pyiqa.create_metric("musiq", device=device),
        "NIQE": pyiqa.create_metric("niqe", device=device),
    }
    return fr_metrics, nr_metrics


# ======================================================================================
# Main: official Restormer test pipeline + eval-style metrics/results
# ======================================================================================


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("restormer_test")

    # ======= Load yaml: keep official Restormer logic =======
    yaml_file = "Deraining/Options/Deraining_Restormer_test.yml"
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode="r"), Loader=Loader)
    x["network_g"].pop("type", None)

    # ======= Create model: keep official Restormer logic =======
    model_restoration = Restormer(**x["network_g"])

    checkpoint = torch.load(args.weights, map_location="cpu")
    state_dict = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    model_restoration.load_state_dict(state_dict)
    print("===>Testing using weights: ", args.weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_restoration = model_restoration.to(device)
    if torch.cuda.is_available():
        model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    fr_metrics, nr_metrics = init_metrics(device)
    use_fid = HAS_FID and not args.no_fid
    if use_fid:
        print("[FID] 已启用：有 GT/target 时会计算 FID。")
    else:
        reason = "命令行传入了 --no_fid" if args.no_fid else f"torchmetrics/FID 依赖不可用: {FID_IMPORT_ERROR}"
        print(f"[FID] 已跳过：{reason}")

    factor = 8
    datasets = ["Rain100L", "Rain100H", "Test100", "Test1200", "Test2800"]
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_root = args.result_dir or FIXED_RESULTS_ROOT

    print(f"[基础保存根目录] {base_results_root}")
    print(f"[模型类型] {MODEL_TYPE}")
    print(f"[运行时间] {run_time}")

    for dataset in datasets:
        inp_dir = os.path.join(args.input_dir, "test", dataset, "input")
        files = natsorted(glob(os.path.join(inp_dir, "*.png")) + glob(os.path.join(inp_dir, "*.jpg")))
        if args.limit is not None:
            files = files[:args.limit]

        total = len(files)
        if total == 0:
            print(f"[跳过] {dataset}: 没有找到输入图片：{inp_dir}")
            continue

        target_dir = find_target_dir(inp_dir, dataset)
        has_target_dir = target_dir is not None
        if has_target_dir:
            print(f"[{dataset}] target_dir: {target_dir}")
        else:
            print(f"[{dataset}] 未找到 target/gt 目录，只计算无参考指标，PSNR/SSIM/LPIPS/FID 为 nan。")

        save_root = build_run_save_root(base_results_root, MODEL_TYPE, dataset, run_time)
        dataset_dir, output_dir, combined_dir, metrics_txt, metrics_csv, metrics_json = create_init_result_files(
            save_root, dataset, total, OUTPUT_KEYS
        )
        print(f"[当前数据集结果目录] {dataset_dir}")

        metric_sums, metric_counts, output_counts = init_metric_storage(OUTPUT_KEYS)
        fid_by_output = OrderedDict()
        fid_counts = OrderedDict()
        test_times: List[float] = []
        processed = 0

        write_results(
            metrics_txt,
            metrics_csv,
            metrics_json,
            dataset,
            processed,
            total,
            OUTPUT_KEYS,
            compute_averages(metric_sums, metric_counts),
            metric_counts,
            output_counts,
            float("nan"),
            final=False,
            fid_counts=fid_counts,
        )

        with torch.no_grad():
            for file_ in tqdm(files, desc=f"Testing {dataset}", unit="img", ncols=100):
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()

                img_name = os.path.splitext(os.path.split(file_)[-1])[0]

                # ======= Official Restormer preprocessing =======
                lq_rgb_uint8 = utils.load_img(file_)
                lq_img_bgr = rgb_uint8_to_bgr_uint8(lq_rgb_uint8)

                img = np.float32(lq_rgb_uint8) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)
                input_ = img.unsqueeze(0).to(device)

                # Padding in case images are not multiples of 8
                h, w = input_.shape[2], input_.shape[3]
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh = H - h if h % factor != 0 else 0
                padw = W - w if w % factor != 0 else 0
                input_ = F.pad(input_, (0, padw, 0, padh), "reflect")

                tic = time.time()
                restored = model_restoration(input_)
                elapsed = time.time() - tic
                test_times.append(elapsed)

                # Unpad images to original dimensions
                restored = restored[:, :, :h, :w]
                restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                # ======= Official Restormer output saving =======
                restored_uint8_rgb = img_as_ubyte(restored)
                utils.save_img(os.path.join(output_dir, img_name + ".png"), restored_uint8_rgb)

                # ======= Eval-style metrics / combined / TXT-CSV-JSON =======
                sr_img_bgr = rgb_uint8_to_bgr_uint8(restored_uint8_rgb)
                outputs = OrderedDict({"final": sr_img_bgr})

                gt_path = find_gt_path(file_, target_dir)
                gt_img_bgr = None
                need_gt = gt_path is not None
                if need_gt:
                    gt_rgb_uint8 = utils.load_img(gt_path)
                    gt_img_bgr = rgb_uint8_to_bgr_uint8(gt_rgb_uint8)

                save_combined(
                    lq_img_bgr,
                    gt_img_bgr,
                    outputs,
                    os.path.join(combined_dir, f"{img_name}_combined.png"),
                )

                gt_rgb_for_fid = bgr_uint8_to_rgb_tensor01(gt_img_bgr, device) if need_gt and gt_img_bgr is not None else None

                for output_key, sr_img in outputs.items():
                    ensure_output_key(output_key, metric_sums, metric_counts, output_counts)
                    output_counts[output_key] += 1

                    metrics = compute_one_output_metrics(
                        sr_img=sr_img,
                        gt_img=gt_img_bgr,
                        need_gt=need_gt,
                        crop_border=args.crop_border,
                        fr_metrics=fr_metrics,
                        nr_metrics=nr_metrics,
                        device=device,
                        logger=logger,
                        img_name=img_name,
                        output_key=output_key,
                    )
                    for metric, value in metrics.items():
                        add_metric(metric_sums, metric_counts, output_key, metric, value)

                    if use_fid and need_gt and gt_rgb_for_fid is not None:
                        try:
                            if output_key not in fid_by_output:
                                fid_by_output[output_key] = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
                                fid_counts[output_key] = 0
                            sr_rgb = bgr_uint8_to_rgb_tensor01(sr_img, device)
                            fid_by_output[output_key].update(gt_rgb_for_fid, real=True)
                            fid_by_output[output_key].update(sr_rgb, real=False)
                            fid_counts[output_key] += 1
                        except Exception as exc:
                            logger.warning("FID update failed for %s/%s: %s", img_name, output_key, exc)

                processed += 1

                if args.write_every > 0 and processed % args.write_every == 0:
                    write_results(
                        metrics_txt,
                        metrics_csv,
                        metrics_json,
                        dataset,
                        processed,
                        total,
                        OUTPUT_KEYS,
                        compute_averages(metric_sums, metric_counts),
                        metric_counts,
                        output_counts,
                        float(np.mean(test_times)) if test_times else float("nan"),
                        final=False,
                        fid_counts=fid_counts,
                    )
                    print(f"[已更新结果文件] processed={processed}: {metrics_txt}")

        fid_values = {}
        if use_fid:
            for output_key, fid_calc in fid_by_output.items():
                try:
                    fid_values[output_key] = float(fid_calc.compute().item())
                except Exception as exc:
                    logger.warning("FID computation failed for %s: %s", output_key, exc)
                    fid_values[output_key] = float("nan")

        averages = compute_averages(metric_sums, metric_counts, fid_values=fid_values)
        avg_time = float(np.mean(test_times)) if test_times else float("nan")
        write_results(
            metrics_txt,
            metrics_csv,
            metrics_json,
            dataset,
            processed,
            total,
            OUTPUT_KEYS,
            averages,
            metric_counts,
            output_counts,
            avg_time,
            final=True,
            fid_counts=fid_counts,
        )

        print("\n" + "=" * 100)
        print(f"FINAL AVERAGE RESULTS ({dataset}, {processed}/{total} samples)")
        print("=" * 100)
        for output_key, values in averages.items():
            print(f"\nOUTPUT_{output_key}")
            for metric in SUMMARY_METRICS:
                v = values.get(metric, float("nan"))
                print(f"  {metric + '(' + direction_symbol(metric) + ')':<16} {'nan' if math.isnan(v) else f'{v:.6f}':>14}")
        print("\nResults saved to:")
        print(f"TXT : {metrics_txt}")
        print(f"CSV : {metrics_csv}")
        print(f"JSON: {metrics_json}")
        print(f"OUTPUT_DIR  : {output_dir}")
        print(f"COMBINED_DIR: {combined_dir}")


if __name__ == "__main__":
    main()
