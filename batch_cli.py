#!/usr/bin/env python3
"""Batch CLI for SBS 2D->3D using DepthAnythingV2 + SBS processing.

Usage:
  python3 batch_cli.py --input-dir input_images --output-dir output --model depth_anything_v2_vitl_fp16.safetensors \
    --depthmap-input-scale 0.75 --sbs-method mesh_warping --sbs-mode parallel --sbs-depth-scale 40 --sbs-depth-blur-strength 7

This script loads the depth model once, iterates over image files in the input directory,
generates depth maps (optionally downscaling input for the depth model) and produces SBS images.
"""
import argparse
import os
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
import matplotlib as mpl

from depth_anything_v2.dpt import DepthAnythingV2
from sbs.sbs import process_image_sbs

# Model configs (copied from run_gradio/main)
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

AVAILABLE_MODELS = [
    'depth_anything_v2_vits_fp16.safetensors',
    'depth_anything_v2_vits_fp32.safetensors',
    'depth_anything_v2_vitb_fp16.safetensors',
    'depth_anything_v2_vitb_fp32.safetensors',
    'depth_anything_v2_vitl_fp16.safetensors',
    'depth_anything_v2_vitl_fp32.safetensors'
]


def load_model(model_name, device, models_dir='models/depthanything'):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from: {AVAILABLE_MODELS}")

    print(f"Selected model: {model_name}")
    dtype = torch.float16 if "fp16" in model_name else torch.float32
    encoder = 'vitl'
    if "vitl" in model_name:
        encoder = "vitl"
    elif "vitb" in model_name:
        encoder = "vitb"
    elif "vits" in model_name:
        encoder = "vits"

    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading {model_name} to {model_path}...")
        os.makedirs(models_dir, exist_ok=True)
        hf_hub_download(repo_id="yushan777/DepthAnythingV2", filename=model_name, local_dir=models_dir, local_dir_use_symlinks=False)

    print(f"Loading model from: {model_path}")
    state_dict = load_safetensors(model_path, device='cpu')

    max_depth = 20.0 if "hypersim" in model_name else 80.0
    is_metric = 'metric' in model_name

    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**{**config, 'is_metric': is_metric, 'max_depth': max_depth})
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device=device, dtype=dtype)
    print("Model loaded successfully.")
    return model, dtype, is_metric


def process_depthmap_image(model, image_tensor, device, dtype, is_metric, output_filename_base, output_dir_frames, write_depthmap=False):
    orig_H, orig_W = image_tensor.shape[2:]
    new_H, new_W = orig_H, orig_W
    if new_W % 14 != 0:
        new_W = new_W - (new_W % 14)
    if new_H % 14 != 0:
        new_H = new_H - (new_H % 14)

    if new_H != orig_H or new_W != orig_W:
        print(f"Resizing input from {orig_W}x{orig_H} to {new_W}x{new_H} to be multiple of 14")
        image_tensor = F.interpolate(image_tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)

    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        depth = model(image_tensor)

    end_time = time.time()
    print(f"Inference took {end_time - start_time:.2f} seconds")

    depth = depth.squeeze(0).squeeze(0)
    depth = (depth - depth.min()) / (depth.max() - depth.min())

    final_H = (orig_H // 2) * 2
    final_W = (orig_W // 2) * 2
    if depth.ndim == 2 and (depth.shape[0] != final_H or depth.shape[1] != final_W):
        depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(final_H, final_W), mode="bilinear", align_corners=False).squeeze()

    depth = torch.clamp(depth, 0, 1)

    if is_metric:
        depth = 1.0 - depth

    depth_np = depth.cpu().numpy()
    depth_visual = (depth_np * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_visual)

    if write_depthmap:
        os.makedirs(output_dir_frames, exist_ok=True)
        grayscale_path = os.path.join(output_dir_frames, f"{output_filename_base}_depth.png")
        depth_image.save(grayscale_path)
        print(f"Saved grayscale depth map to: {grayscale_path}")

    return depth_image


def generate_sbs_image_from_depth(original_input_image, depth_map_pil, model_name, sbs_method, sbs_depth_scale, sbs_mode, sbs_depth_blur_strength):
    if original_input_image is None or depth_map_pil is None or model_name is None:
        print("Missing image, depth map, or model name for SBS generation.")
        return None

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU detected for SBS. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon GPU detected for SBS. Using MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU detected for SBS. Using CPU.")

    dtype = torch.float16 if "fp16" in model_name else torch.float32

    try:
        if original_input_image.mode != 'RGB':
            original_input_image = original_input_image.convert('RGB')

        transform_to_tensor = transforms.ToTensor()
        base_image_for_sbs = transform_to_tensor(original_input_image).permute(1, 2, 0).unsqueeze(0)
        base_image_for_sbs = base_image_for_sbs.to(device=device, dtype=torch.float16)

        depth_map_for_sbs = transform_to_tensor(depth_map_pil).permute(1, 2, 0).unsqueeze(0)
        depth_map_for_sbs = depth_map_for_sbs.to(device=device, dtype=torch.float16)

        if sbs_depth_blur_strength % 2 == 0:
            sbs_depth_blur_strength += 1
            print(f"SBS Depth Blur Strength adjusted to {sbs_depth_blur_strength} (must be odd-numbered).")

        sbs_image_tensor = process_image_sbs(
            base_image=base_image_for_sbs,
            depth_map=depth_map_for_sbs,
            method=sbs_method,
            depth_scale=sbs_depth_scale,
            mode=sbs_mode,
            depth_blur_strength=sbs_depth_blur_strength,
        )

        sbs_image_pil = transforms.ToPILImage()(sbs_image_tensor.squeeze(0).cpu().permute(2, 0, 1))
        return sbs_image_pil
    except Exception as e:
        print(f"Error generating SBS image: {e}")
        return None


def find_image_files(input_dir, recursive=False):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    results = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(input_dir):
            rel_dir = os.path.relpath(dirpath, input_dir)
            if rel_dir == '.':
                rel_dir = ''
            for f in sorted(filenames):
                if os.path.splitext(f.lower())[1] in exts:
                    results.append((os.path.join(dirpath, f), rel_dir))
    else:
        for f in sorted(os.listdir(input_dir)):
            full = os.path.join(input_dir, f)
            if os.path.isfile(full) and os.path.splitext(f.lower())[1] in exts:
                results.append((full, ''))
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch SBS CLI")
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', default=AVAILABLE_MODELS[4] if len(AVAILABLE_MODELS) > 4 else AVAILABLE_MODELS[0], choices=AVAILABLE_MODELS)
    parser.add_argument('--models-dir', default='models/depthanything')
    parser.add_argument('--depthmap-input-scale', type=float, default=0.75)
    parser.add_argument('--recursive', action='store_true', help='Recursively traverse input directory and preserve subdirectory structure in output-dir')
    parser.add_argument('--write-depthmap', action='store_true', help='Write grayscale depth map files to the output directory')
    parser.add_argument('--write-depthmap-only', action='store_true', help='Only write depth map files and skip generating SBS images')
    parser.add_argument('--sbs-method', choices=['mesh_warping', 'grid_sampling'], default='mesh_warping')
    parser.add_argument('--sbs-mode', choices=['parallel', 'cross-eyed'], default='parallel')
    parser.add_argument('--sbs-depth-scale', type=int, default=40)
    parser.add_argument('--sbs-depth-blur-strength', type=int, default=7)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device for depth model
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA GPU detected. Using GPU for depth model.')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Apple Silicon GPU detected. Using MPS for depth model.')
    else:
        device = torch.device('cpu')
        print('No GPU detected. Using CPU for depth model.')

    # Load model once
    depth_model, dtype, is_metric = load_model(args.model, device, args.models_dir)

    files = find_image_files(args.input_dir, recursive=args.recursive)
    if not files:
        print('No image files found in input directory.')
        return

    transform_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for img_path, rel_dir in files:
        print(f"Processing {img_path}...")
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # determine output subdirectory and create it
        out_subdir = args.output_dir if not rel_dir else os.path.join(args.output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)

        # create working copy and downscale for depth model if requested
        image_for_depth_processing = img.copy()
        if args.depthmap_input_scale < 1.0:
            ow, oh = image_for_depth_processing.size
            nw = max(1, int(ow * args.depthmap_input_scale))
            nh = max(1, int(oh * args.depthmap_input_scale))
            print(f"Downscaling for depth: {ow}x{oh} -> {nw}x{nh}")
            image_for_depth_processing = image_for_depth_processing.resize((nw, nh), Image.Resampling.BICUBIC)

        image_tensor = transform_normalize(image_for_depth_processing).unsqueeze(0).to(device=device, dtype=dtype)

        write_map = args.write_depthmap or args.write_depthmap_only

        depth_pil = process_depthmap_image(
            depth_model,
            image_tensor,
            device,
            dtype,
            is_metric,
            base_name,
            out_subdir,
            write_depthmap=write_map,
        )

        if args.write_depthmap_only:
            print(f"--write-depthmap-only set; skipping SBS generation for {img_path}.")
            continue

        sbs_pil = generate_sbs_image_from_depth(img, depth_pil, args.model, args.sbs_method, args.sbs_depth_scale, args.sbs_mode, args.sbs_depth_blur_strength)

        if sbs_pil is not None:
            sbs_out = os.path.join(out_subdir, f"{base_name}_sbs.png")
            sbs_pil.save(sbs_out)
            print(f"Saved SBS image to: {sbs_out}")
        else:
            print(f"Failed to generate SBS for {img_path}")

    print("Batch processing complete.")


if __name__ == '__main__':
    main()
