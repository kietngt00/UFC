import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
from pycocotools.coco import COCO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from annotator.hed import HEDdetector
from annotator.midas import MidasDetector
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.sketch import SketchDetector
from annotator.openpose import OpenposeDetector

hed_annotator = HEDdetector()
midas_annotator = MidasDetector()
canny_annotator = CannyDetector()
mlsd_annotator = MLSDdetector()
sketch_annotator = SketchDetector()
pose_annotator = OpenposeDetector()
# NOTE: Densepose is extracted seperatedly

"""
    This file is only used to extract the condition from the generated images.
    Generated images Dir Structure:
    ├── task1
    │   ├── images
    │   ├── annotations
    ├── task2
    │   ├── images
    │   ├── annotations  
    ...  
"""

def openpose_to_coco_keypoints(openpose_kpts):
    openpose_to_coco = [
        0,   # Nose
        15,  # LEye
        14,  # REye
        17,  # LEar
        16,  # REar
        5,   # LShoulder
        2,   # RShoulder
        6,   # LElbow
        3,   # RElbow
        7,   # LWrist
        4,   # RWrist
        11,  # LHip
        8,   # RHip
        12,  # LKnee
        9,   # RKnee
        13,  # LAnkle
        10   # RAnkle
    ]

    coco_kpts = []
    for idx in openpose_to_coco:
        if openpose_kpts[idx] is not None:
            x, y, v = openpose_kpts[idx]
        else:
            x, y, v = 0, 0, 0
        coco_kpts.extend([float(x), float(y), int(v)])
    
    return coco_kpts


def to_coco_format(candidate, subset, image_id):
    annotations = []
    if not subset:
        return annotations
    for person in subset:
        keypoints = []
        score = person[-2]  # total score for this person
        for i in range(18):  # for each keypoint
            index = int(person[i])
            if index == -1:
                keypoints.append([0, 0, 0])  # Not found
            else:
                x, y, conf = candidate[index][:3]
                keypoints.append([float(x), float(y), 2 if conf > 0.05 else 1])  # visibility: 2 = labeled & visible
        keypoints = openpose_to_coco_keypoints(keypoints)
        annotations.append({
            "image_id": image_id,
            "category_id": 1,
            "keypoints": keypoints,
            "score": float(score)
        })
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    if args.task == 'pose':
        json_path = "/data2/kietngt00/coco2017/annotations/keypoints_captions_val.json"
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        filename_to_id = {item['file_name']: item['image_id'] for item in coco_data}

    results = []

    filenames = glob(args.path + f'/{args.task}/fid/*.jpg')   
    args.path = os.path.join(args.path, args.task, "annotations")
    os.makedirs(args.path, exist_ok=True)
    valid_count = 0
    for file in tqdm(filenames):
        name = file.split('/')[-1]
        save_path = Path(args.path, name)

        image = Image.open(file)

        if args.task == 'hed':
            image_annotated = hed_annotator(np.array(image))
            Image.fromarray(image_annotated).save(save_path)
        elif args.task == 'canny':
            image_annotated = canny_annotator(np.array(image), 100, 200)
            Image.fromarray(image_annotated).save(save_path)
        elif args.task == 'mlsd':
            image_annotated = mlsd_annotator(np.array(image),0.1, 0.1)
            Image.fromarray(image_annotated).save(save_path)
        elif args.task == 'sketch':
            image_annotated = sketch_annotator(np.array(image))
            Image.fromarray(image_annotated).save(save_path)
        elif args.task == 'pose':
            image_id = filename_to_id.get(name)
            print(f"Processing image {name} with ID {image_id}")
            if image_id is not None:
                image_annotated, result_dict = pose_annotator(np.array(image))
                Image.fromarray(image_annotated).save(save_path)
                annotation = to_coco_format(result_dict['candidate'], result_dict['subset'], image_id)

                if annotation:  
                    results.extend(annotation)
                    print(f"Valid annotations for image {name}: {len(annotation)}")
                    valid_count += len(annotation)
                else:
                    print(f"Warning: No valid annotations for image {name}")
            else:
                print(f"Image ID not found for {name}. Skipping annotation.")
                continue
            
            

        elif args.task == 'depth':
            depth, _ = midas_annotator(np.array(image))
            Image.fromarray(depth).save(save_path)
        elif args.task == 'normal':
            _, normal = midas_annotator(np.array(image))
            Image.fromarray(normal).save(save_path)
        else:
            raise ValueError(f"Task {args.task} not recognized")
    
    if args.task == 'pose':
        print("Total valid annotations:", valid_count)
        
        results.sort(key=lambda ann: ann['image_id'])
        with open(Path(args.path, 'annotations.json'), 'w') as f:
            json.dump(results, f)
            print(f"Saved annotations to {Path(args.path, 'annotations.json')}")
