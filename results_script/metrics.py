import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import lpips
import torch

# Required functions to calculate metrics

def compute_lpips(image1, image2):
    loss_fn = lpips.LPIPS(net='alex')
    image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return loss_fn(image1, image2).item()

def compute_composite_score(psnr_value, ssim_value, lpips_value, weights):
    normalized_psnr = psnr_value / 35.0
    composite_score = (weights['psnr'] * normalized_psnr +
                       weights['ssim'] * ssim_value +
                       weights['lpips'] * (1 - lpips_value))
    return composite_score

weights = {
    'psnr': 0.2,
    'ssim': 0.35,
    'lpips': 0.45
}

### DATA OBTAINED HERE
ground_truth_folder = '/home/student/Desktop/results/stage1/scene_738/test_gt'
rendered_folder = '/home/student/Desktop/results/stage1/scene_738/test_true_depth_render'
output_file = '/home/student/Desktop/results/stage1/scene_738/ssim_psnr_lpips_composite_true_depth_resultss.txt'
excel_output_file = '/home/student/Desktop/results/stage1/scene_738/ssim_psnr_lpips_composite_true_depth_resultss.xlsx'


os.makedirs(os.path.dirname(output_file), exist_ok=True)


ground_truth_images = sorted(os.listdir(ground_truth_folder))
rendered_images = sorted(os.listdir(rendered_folder))


if len(ground_truth_images) != len(rendered_images):
    raise ValueError("The number of ground truth and rendered images do not match.")


results = []

with open(output_file, 'w') as f:
    f.write("Image, SSIM, PSNR, LPIPS, Composite Score\n")

    for img_name in ground_truth_images:
        gt_img_path = os.path.join(ground_truth_folder, img_name)
        rend_img_path = os.path.join(rendered_folder, img_name)

        gt_img = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
        rend_img = cv2.imread(rend_img_path, cv2.IMREAD_COLOR)

        if gt_img is None or rend_img is None:
            raise ValueError(f"Could not read images: {gt_img_path}, {rend_img_path}")

        gt_img_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
        rend_img_gray = cv2.cvtColor(rend_img, cv2.COLOR_BGR2GRAY)

        ssim_value = ssim(gt_img_gray, rend_img_gray)
        psnr_value = psnr(gt_img, rend_img)
        lpips_value = compute_lpips(gt_img, rend_img)

        composite_score = compute_composite_score(psnr_value, ssim_value, lpips_value, weights)
        f.write(f"{img_name}, {ssim_value}, {psnr_value}, {lpips_value}, {composite_score}\n")

        results.append([img_name, ssim_value, psnr_value, lpips_value, composite_score])

df = pd.DataFrame(results, columns=['Image', 'SSIM', 'PSNR', 'LPIPS', 'Composite Score'])
df.to_excel(excel_output_file, index=False, header=True)

print(f"All values have been written to {output_file} and {excel_output_file}")

