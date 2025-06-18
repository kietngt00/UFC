import numpy as np
from PIL import Image
import torch
import argparse
import cv2
from tqdm import tqdm

def compute_batch_miou(predictions, groundtruths, class_ids):
    
    iou_scores = {c: [] for c in class_ids}
    fg_bg_iou_scores = {'foreground': [], 'background': []}
    common_keys = sorted(set(predictions.keys()) & set(groundtruths.keys()))

    for key in common_keys:

        for c in class_ids:
            pred_mask = predictions[key] == c
            gt_mask = groundtruths[key] == c

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            if union == 0:
                if gt_mask.sum() > 0:  
                    iou_scores[c].append(0.0)
                else:
                    iou_scores[c].append(float("nan"))
            else:
                iou_scores[c].append(intersection / union)

        pred_fg_mask = predictions[key] != 0
        gt_fg_mask = groundtruths[key] != 0
        fg_intersection = np.logical_and(pred_fg_mask, gt_fg_mask).sum()
        fg_union = np.logical_or(pred_fg_mask, gt_fg_mask).sum()
        if fg_union == 0:
            if gt_fg_mask.sum() > 0:
                fg_bg_iou_scores['foreground'].append(0.0)
            else:
                fg_bg_iou_scores['foreground'].append(float("nan"))
        else:
            fg_bg_iou_scores['foreground'].append(fg_intersection / fg_union)

        # 배경 (클래스 ID == 0) IoU 계산
        pred_bg_mask = predictions[key] == 0
        gt_bg_mask = groundtruths[key] == 0
        bg_intersection = np.logical_and(pred_bg_mask, gt_bg_mask).sum()
        bg_union = np.logical_or(pred_bg_mask, gt_bg_mask).sum()
        if bg_union == 0:
            if gt_bg_mask.sum() > 0:
                fg_bg_iou_scores['background'].append(0.0)
            else:
                fg_bg_iou_scores['background'].append(float("nan"))
        else:
            fg_bg_iou_scores['background'].append(bg_intersection / bg_union)


    mean_iou_per_class = {c: np.nanmean(iou_scores[c]) for c in class_ids}

    overall_miou = np.nanmean(list(mean_iou_per_class.values()))

    mean_fg_bg_iou = {
        'foreground': np.nanmean(fg_bg_iou_scores['foreground']),
        'background': np.nanmean(fg_bg_iou_scores['background'])
    }

    return mean_iou_per_class, overall_miou, mean_fg_bg_iou

def visualize(segmap, mask, matrix, bbox_xyxy):
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return segmap
    mask, matrix = _resize(mask, matrix, w, h)
    segmap[y1:y2, x1:x2] = matrix
    return segmap

def _resize(mask, matrix, w, h):
    if (w != mask.shape[1]) or (h != mask.shape[0]):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if (w != matrix.shape[1]) or (h != matrix.shape[0]):
        matrix = cv2.resize(matrix, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask, matrix

def get_segmentation_mask(result_path):
    with open(result_path, 'rb') as f:
        data = torch.load(f, map_location='cpu', weights_only=False)
    labels = dict()
    error_count = 0
    error_files = []
    for i in tqdm(range(len(data))):
        file_name = data[i]['file_name']
        
        segmap = np.zeros((512, 512), dtype=np.uint8)
        idx = file_name.split('/')[-1].split('.')[0].split('_')[-1]
        idx = int(idx)
        if data[i].get('pred_densepose') is None:
            labels[idx] = segmap
            error_count += 1
            error_files.append((idx, file_name))
            continue
        for j in range(len(data[i]['pred_densepose'])):
            bbox_xyxy = data[i]['pred_boxes_XYXY'][j].numpy()
            segm = data[i]['pred_densepose'][j].labels.numpy()
            matrix = segm.astype(np.uint8)
            mask = np.zeros(matrix.shape, dtype=np.uint8)
            mask[segm > 0] = 1
            segmap = visualize(segmap, mask, matrix, bbox_xyxy)
        labels[idx] = segmap
    print(f'Error count: {error_count}')
    if error_count > 0:
        print(f'Files with detection errors (idx, file_name): {error_files}')
    return labels

def main(predict_path, gt_path, num_classes):
    predict_labels = get_segmentation_mask(predict_path)
    gt_labels = get_segmentation_mask(gt_path)
    
    missing_preds = sorted(set(gt_labels.keys()) - set(predict_labels.keys()))
    missing_gts = sorted(set(predict_labels.keys()) - set(gt_labels.keys()))
    if missing_preds:
        print(f"Missing in predictions: {len(missing_preds)} samples, keys: {missing_preds}")
    if missing_gts:
        print(f"Missing in groundtruths: {len(missing_gts)} samples, keys: {missing_gts}")
    
    class_ids = list(range(num_classes))  
    mean_iou_per_class, overall_miou, mean_fg_bg_iou = compute_batch_miou(predict_labels, gt_labels, class_ids)
    
    print("\nClass-wise mIoU:")
    for c, iou in sorted(mean_iou_per_class.items()):
        print(f"Class {c}: {iou:.4f}")
    print(f"\nForeground mIoU: {mean_fg_bg_iou['foreground']:.4f}")
    print(f"Background mIoU: {mean_fg_bg_iou['background']:.4f}")
    print(f"Overall mIoU: {overall_miou:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DensePose segmentation with mIoU')
    parser.add_argument('--predict_path', type=str, required=True, help='Path to prediction results (.pt file)')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth results (.pt file)')
    parser.add_argument('--num_classes', type=int, default=25, required=False, help='Number of classes (including background)')
    args = parser.parse_args()
    main(args.predict_path, args.gt_path, args.num_classes)