import os
from math import log10, sqrt
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd

import pywt
import bm3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


class SimulateFluoroscopy:
    """
    Transforms a high-quality cine frame into a low-quality fluoroscopy frame.
    Assumes input is a PyTorch tensor of shape (C, H, W) with values in [0.0, 1.0].
    """
    def __init__(self,
                 downsample_scale=(0.5, 0.75),
                 blur_kernel_range=(3, 5),
                 photon_count_range=(50, 500),
                 gaussian_std_range=(0.005, 0.02)):
        self.downsample_scale = downsample_scale
        self.blur_kernel_range = blur_kernel_range
        self.photon_count_range = photon_count_range
        self.gaussian_std_range = gaussian_std_range

    def __call__(self, cine_img):
        _, h, w = cine_img.shape
        fluoro_img = cine_img.clone()

        # 1. Spatial Degradation
        scale = random.uniform(*self.downsample_scale)
        new_h, new_w = int(h * scale), int(w * scale)
        fluoro_img = fluoro_img.unsqueeze(0)
        fluoro_img = F.interpolate(fluoro_img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        fluoro_img = F.interpolate(fluoro_img, size=(h, w), mode='bicubic', align_corners=False)
        fluoro_img = fluoro_img.squeeze(0)

        # 2. Blur
        kernel_size = random.choice(range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2))
        sigma = random.uniform(0.5, 1.5)
        fluoro_img = TF.gaussian_blur(fluoro_img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        # 3. Poisson Noise
        photon_count = random.uniform(*self.photon_count_range)
        fluoro_img = fluoro_img * photon_count
        fluoro_img = torch.poisson(torch.clamp(fluoro_img, min=0.0))
        fluoro_img = fluoro_img / photon_count

        # 4. Gaussian Noise
        std = random.uniform(*self.gaussian_std_range)
        noise = torch.randn_like(fluoro_img) * std
        fluoro_img = fluoro_img + noise

        return torch.clamp(fluoro_img, 0.0, 1.0)


def safe_bm3d_denoise(image, estimated_sigma_8bit):
    img_float = image.astype(np.float32) / 255.0
    sigma_float = estimated_sigma_8bit / 255.0

    if len(image.shape) == 3:
        img_rgb = cv.cvtColor(img_float, cv.COLOR_BGR2RGB)
        denoised = bm3d.bm3d(img_rgb, sigma_psd=sigma_float, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        denoised = cv.cvtColor(denoised, cv.COLOR_RGB2BGR)
    else:
        denoised = bm3d.bm3d(img_float, sigma_psd=sigma_float, stage_arg=bm3d.BM3DStages.ALL_STAGES)

    return np.clip(denoised * 255.0, 0, 255).astype(np.uint8)


def calculate_psnr(original, compressed, title):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr, mse


def process_single_image(input_path, archive_folder, is_vis_image):
    """
    Worker function to process a single image so it can run on an isolated CPU core.
    """
    # Instantiate locally to ensure thread safety
    degradator = SimulateFluoroscopy(photon_count_range=(30, 100))
    
    rel_path = os.path.relpath(input_path, archive_folder)
    dir_path, image_filename = os.path.split(rel_path)
    dir_levels = dir_path.split(os.sep) if dir_path and dir_path != '.' else []

    original_np_raw = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    if original_np_raw is None:
        return None, None, f"Could not read {input_path}"

    original_np_resized = original_np_raw.copy()
    original_tensor = TF.to_tensor(original_np_resized).float()

    # Apply Degradation
    simulated_fluoro_tensor = degradator(original_tensor)
    sim_img = (simulated_fluoro_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    current_images = {
        "Original": original_np_resized,
        "Noisy": sim_img
    }

    # 1. Standard OpenCV Filters
    current_images["Mean"] = cv.blur(sim_img, (5, 5))
    current_images["Bilateral"] = cv.bilateralFilter(sim_img, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. PCA Denoising
    patch_size = (8, 8)
    patches = extract_patches_2d(sim_img, patch_size)
    n_patches, h, w = patches.shape
    patches_flat = patches.reshape(n_patches, h * w)
    pca = PCA(n_components=10)
    compressed_patches = pca.fit_transform(patches_flat)
    denoised_patches_flat = pca.inverse_transform(compressed_patches)
    denoised_patches = denoised_patches_flat.reshape(n_patches, h, w)
    current_images["PCA"] = np.clip(reconstruct_from_patches_2d(denoised_patches, sim_img.shape), 0, 255).astype(np.uint8)

    # 3. Spatial / Fourier Filter
    f_shift = np.fft.fftshift(np.fft.fft2(sim_img))
    rows, cols = sim_img.shape
    mask = np.zeros((rows, cols), np.uint8)
    cv.circle(mask, (cols // 2, rows // 2), 60, 1, thickness=-1)
    f_ishift = np.fft.ifftshift(f_shift * mask)
    current_images["Spatial"] = np.clip(np.abs(np.fft.ifft2(f_ishift)), 0, 255).astype(np.uint8)

    # 4. BM3D
    current_images["BM3D"] = safe_bm3d_denoise(sim_img, estimated_sigma_8bit=20)

    # Format Metrics
    base_identifiers = {"filename": image_filename}
    for i, level_name in enumerate(dir_levels):
        base_identifiers[f"dir_level_{i}"] = level_name
        
    wide_row = base_identifiers.copy()

    for title, img in current_images.items():
        if title != "Original": 
            psnr, mse = calculate_psnr(current_images["Original"], img, title)
            wide_row[f"PSNR_{title}"] = psnr
            wide_row[f"MSE_{title}"] = mse

    vis_state = None
    if is_vis_image:
        vis_state = {"filename": image_filename, "images": current_images}

    return wide_row, vis_state, None


def main():
    # Setup Paths
    ARCHIVE_FOLDER = "./archive" 
    OUTPUT_CSV_WIDE = "./denoising_results_wide.csv"
    OUTPUT_IMAGE_DIR = "./denoised_comparison_images"
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
    all_image_paths = []
    max_dir_depth = 0
    
    # 1. Collect paths and establish maximum directory depth for strict CSV columns
    for root, _, files in os.walk(ARCHIVE_FOLDER):
        for file in files:
            if file.lower().endswith(valid_extensions):
                all_image_paths.append(os.path.join(root, file))
                
                # Calculate depth
                rel = os.path.relpath(root, ARCHIVE_FOLDER)
                depth = len(rel.split(os.sep)) if rel != '.' else 0
                max_dir_depth = max(max_dir_depth, depth)

    if not all_image_paths:
        raise FileNotFoundError(f"No image files found in {ARCHIVE_FOLDER} or its subdirectories.")

    num_comparison_images = min(7, len(all_image_paths))
    images_to_visualize = set(random.sample(all_image_paths, num_comparison_images))
    vis_image_states = []

    # 2. Pre-generate strict CSV column structures (now including 'Noisy')
    csv_methods = ["Noisy", "Mean", "Bilateral", "PCA", "Spatial", "BM3D"]
    
    wide_cols = ["filename"] + [f"dir_level_{i}" for i in range(max_dir_depth)]
    for m in csv_methods:
        wide_cols.extend([f"PSNR_{m}", f"MSE_{m}"])

    # Initialize empty CSV with headers
    pd.DataFrame(columns=wide_cols).to_csv(OUTPUT_CSV_WIDE, index=False)

    print(f"--- Multiprocessing {len(all_image_paths)} images across available CPU cores ---")

    # 3. Multiprocessing Pool
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_single_image, 
                input_path, 
                ARCHIVE_FOLDER, 
                input_path in images_to_visualize
            ): input_path for input_path in all_image_paths
        }

        # 4. Checkpointing (Active Saving)
        for i, future in enumerate(as_completed(futures), 1):
            wide_row, vis_state, error = future.result()
            
            if error:
                print(f"[{i}/{len(all_image_paths)}] Warning: {error}")
                continue

            # Reindex ensures columns align perfectly with headers even if directory depths vary
            wide_df = pd.DataFrame([wide_row]).reindex(columns=wide_cols)
            wide_df.to_csv(OUTPUT_CSV_WIDE, mode='a', header=False, index=False)

            if vis_state:
                vis_image_states.append(vis_state)

            print(f"[{i}/{len(all_image_paths)}] Completed: {wide_row['filename']}")

    print(f"\nProcessing Complete! Wide metrics saved to {OUTPUT_CSV_WIDE}.")

    # --- Generate 7x7 Comparison Plot ---
    if not vis_image_states:
        return

    # Method titles for the plot
    plot_methods = ["Original", "Noisy", "Mean", "Bilateral", "PCA", "Spatial", "BM3D"]

    fig, axes = plt.subplots(len(plot_methods), num_comparison_images, 
                             figsize=(num_comparison_images * 3, len(plot_methods) * 3))
    
    if num_comparison_images == 1:
        axes = np.array([axes]).T 
    elif len(plot_methods) == 1:
        axes = np.array([axes])

    for col_idx, img_state in enumerate(vis_image_states):
        filename = img_state["filename"]
        images = img_state["images"]
        
        for row_idx, method_title in enumerate(plot_methods):
            ax = axes[row_idx, col_idx]
            img = images.get(method_title, np.zeros_like(images["Original"])) 
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if col_idx == 0: 
                ax.set_ylabel(method_title, rotation=90, size=10, weight='bold')
            if row_idx == 0: 
                ax.set_title(f"Img: {filename}", fontsize=10)

    plt.suptitle('Comparison of Classical Denoising Methods Across Selected Images', y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    plt.savefig(os.path.join(OUTPUT_IMAGE_DIR, "denoising_comparison_grid.png"))
    plt.show()

if __name__ == "__main__":
    main()
