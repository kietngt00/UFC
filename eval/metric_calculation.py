import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
from skimage.metrics import structural_similarity as ssim
import torch
import os
from collections import Counter
from torch.nn import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# TODO: debug after having generated images
# NOTE: mIoU for densepose is computed seperately

def main(args):
    if args.task in ['canny', 'mlsd', 'hed', 'sketch']:
        filenames = glob(args.gen_path + f'/{args.task}/annotations/*.jpg')
        results = []
        for filename in tqdm(filenames):
            file_id = filename.split("/")[-1]
            gen_label = np.array(Image.open(filename))
            gt_label = np.array(Image.open(args.gt_path + f"/{args.task}/{file_id}"))
            result = ssim(gen_label, gt_label, 
                          gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                          data_range=gt_label.max() - gt_label.min())
            results.append(result)
        mean_result = np.mean(results)
        print(f"Mean SSIM for {args.task}: {mean_result}")
        np.save(args.gen_path + f"/{args.task}/ssim.npy", mean_result)

    elif args.task == 'depth': 
        filenames = glob(args.gen_path + f'/{args.task}/annotations/*.jpg')
        results = []
        for filename in tqdm(filenames):
            file_id = filename.split("/")[-1]
            gen_label = np.array(Image.open(filename))
            gt_label = np.array(Image.open(args.gt_path + f"/{args.task}/{file_id}"))
            result = np.mean((gen_label - gt_label) ** 2)
            results.append(result)
        mean_result = np.mean(results)
        print(f"Mean MSE for {args.task}: {mean_result}")
        np.save(args.gen_path + f"/{args.task}/mse.npy", mean_result)

    elif args.task == 'normal':
        filenames = glob(args.gen_path + f'/{args.task}/annotations/*.jpg')
        results = []
        for filename in tqdm(filenames):
            file_id = filename.split("/")[-1]
            gen_label = np.array(Image.open(filename))
            gt_label = np.array(Image.open(args.gt_path + f"/{args.task}/{file_id}"))
            # Convert to torch tensors and normalize to [0, 1]
            pred_normals = torch.tensor(gen_label).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            gt_normals = torch.tensor(gt_label).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            # Reshape to (N, 3)
            pred_normals = pred_normals.permute(0, 2, 3, 1).reshape(-1, 3)
            gt_normals = gt_normals.permute(0, 2, 3, 1).reshape(-1, 3)

            # Normalize vectors to unit length
            pred_normals = pred_normals / torch.norm(pred_normals, dim=1, keepdim=True)
            gt_normals = gt_normals / torch.norm(gt_normals, dim=1, keepdim=True)

            # Compute dot product and clamp to avoid numerical issues
            dot_product = torch.sum(pred_normals * gt_normals, dim=1).clamp(-1.0, 1.0)

            # Compute angular error in degrees
            angular_error = torch.acos(dot_product) * (180.0 / np.pi)
            results.append(angular_error.mean().item())

        mean_result = np.mean(results)
        print(f"Mean Angular Error for {args.task}: {mean_result}")
        np.save(args.gen_path + f"/{args.task}/mae.npy", mean_result)
    
    elif args.task == 'pose':
        
        coco_gt = COCO("datasets/coco2017/annotations/conv.json")

        predict_path = args.gen_path + f"/annotations/annotations.json"
        
        with open(predict_path, 'r') as f:
            dt = json.load(f)
        
        if isinstance(dt, dict):
            dt = dt.get('annotations', dt)
            
        valid_ids = set(coco_gt.getImgIds())
        dt = [ann for ann in dt if ann['image_id'] in valid_ids]
        coco_dt = coco_gt.loadRes(dt)

        # Initialize COCOeval object
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')

        # Count GT and DT annotations per image
        gt_cnt = Counter(a['image_id'] for a in coco_gt.dataset['annotations'])
        dt_cnt = Counter(a['image_id'] for a in dt)

        # Debug: Print total annotations
        print(f"Total GT annotations: {sum(gt_cnt.values())}")
        print(f"Total DT annotations: {sum(dt_cnt.values())}")
    

        valid_img_ids = [img_id for img_id in gt_cnt if img_id in dt_cnt and dt_cnt[img_id] >= gt_cnt[img_id]]

        coco_eval.params.imgIds = valid_img_ids
        # coco_eval.params.imgIds = coco_gt.getImgIds()

        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="hehe")
    parser.add_argument("--gen_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    main(args)
