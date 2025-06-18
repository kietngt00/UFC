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
from annotator.uniformer import UniformerDetector
from annotator.midas import MidasDetector
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.sketch import SketchDetector
from annotator.openpose import OpenposeDetector, util

hed_annotator = HEDdetector()
midas_annotator = MidasDetector()
canny_annotator = CannyDetector()
mlsd_annotator = MLSDdetector()
sketch_annotator = SketchDetector()
pose_annotator = OpenposeDetector()
# NOTE: Densepose is extracted seperatedly

def estimate_openpose_from_coco(coco_kpts):
    openpose_kpts = [None] * 18

    # Mapping from COCO index → OpenPose index
    coco_to_openpose = {
        0: 0,   # nose
        1: 15,  # left eye
        2: 14,  # right eye
        3: 17,  # left ear
        4: 16,  # right ear
        5: 5,   # left shoulder
        6: 2,   # right shoulder
        7: 6,   # left elbow
        8: 3,   # right elbow
        9: 7,   # left wrist
        10: 4,  # right wrist
        11: 11, # left hip
        12: 8,  # right hip
        13: 12, # left knee
        14: 9,  # right knee
        15: 13, # left ankle
        16: 10  # right ankle
    }

    # Fill known keypoints
    for coco_i, openpose_i in coco_to_openpose.items():
        x, y, v = coco_kpts[coco_i * 3: coco_i * 3 + 3]
        openpose_kpts[openpose_i] = [x, y, v, openpose_i]

    # Estimate neck at index 1
    l_sh = openpose_kpts[5]
    r_sh = openpose_kpts[2]
    if l_sh and r_sh:
        neck_x = (l_sh[0] + r_sh[0]) / 2
        neck_y = (l_sh[1] + r_sh[1]) / 2
        neck_v = min(l_sh[2], r_sh[2])
        openpose_kpts[1] = [neck_x, neck_y, neck_v, 1]
    else:
        openpose_kpts[1] = [0, 0, 0, 1]

    return openpose_kpts


def draw_skeleton(image, coco_keypoints_list):
    candidate = []
    subset = []

    candidate_id = 0    
    for person in coco_keypoints_list:
        coco_kpts = person["keypoints"]
        openpose_kpts = estimate_openpose_from_coco(coco_kpts)
        person_row = -1 * np.ones(20)  # 17 keypoints + score + part count

        part_count = 0
        part_score = 0

        for i in range(18):
            x, y, v, _ = openpose_kpts[i]
            if v > 0:  # Keypoint is labeled
                person_row[i] = candidate_id
                candidate.append([x, y, v, candidate_id])
                candidate_id += 1
                part_count += 1
                part_score += 1.0  # v is used as confidence here (you can change it to 1.0 or some other heuristic)

        person_row[18] = part_score  # total score
        person_row[19] = part_count  # total part count
        subset.append(person_row)

    candidate, subset = np.array(candidate), np.array(subset)
    
    canvas = np.zeros_like(np.array(image))
    canvas = util.draw_bodypose(canvas, candidate, subset)
    return canvas



"""
    This file is only used to extract the condition from COCO validation set.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    coco = COCO("datasets/coco2017/annotations/person_keypoints_val2017.json")

    with open("datasets/coco2017/annotations/keypoints_captions_val.json", 'r') as f:
        keypoints = json.load(f)
    filename_to_id = {keypoint['file_name']: keypoint['image_id'] for keypoint in keypoints}

    filenames = glob(args.path + '/images/*.jpg')
    for task in ['hed', 'canny', 'mlsd', 'sketch', 'depth', 'normal', 'pose']:
        os.makedirs(args.path + f"/{task}", exist_ok=True)

    depth_results = []
    normal_results = []
    for file in tqdm(filenames):
        name = file.split('/')[-1]
        image = Image.open(file).resize((512, 512), Image.LANCZOS).convert("RGB")

        image_hed = hed_annotator(np.array(image))
        Image.fromarray(image_hed).save(Path(args.path, "hed", name))

        image_canny = canny_annotator(np.array(image), 100, 200)
        Image.fromarray(image_canny).save(Path(args.path, "canny", name))

        image_mlsd = mlsd_annotator(np.array(image),0.1, 0.1)
        Image.fromarray(image_mlsd).save(Path(args.path, "mlsd", name))

        image_sketch = sketch_annotator(np.array(image))
        Image.fromarray(image_sketch).save(Path(args.path, "sketch", name))

        image_depth, image_normal = midas_annotator(np.array(image))
        Image.fromarray(image_depth).save(Path(args.path, "depth", name))
        Image.fromarray(image_normal).save(Path(args.path, "normal", name))
        depth_results.append(image_depth) # This save RGB values in range [0, 255]
        normal_results.append(image_normal)  # This save RGB values in range [0, 255]
        
        # pose
        image_id = filename_to_id.get(name)
        if image_id is not None:
            # Get keypoints from coco annotation
            ann_ids = coco.getAnnIds(imgIds=image_id)
            anns = coco.loadAnns(ann_ids)
            keypoints_list = [ann for ann in anns if 'keypoints' in ann]
            # Draw skeleton, reshape the canvas to 512x512
            image = Image.open(file).convert("RGB")
            canvas = draw_skeleton(image, keypoints_list)
            canvas = Image.fromarray(canvas).resize((512, 512), Image.LANCZOS)
            canvas.save(Path(args.path, "pose", name))

    depth_results = np.array(depth_results)
    normal_results = np.array(normal_results)
    np.save(Path(args.path, "depth", 'depth.npy'), depth_results)
    np.save(Path(args.path, "normal", 'normal.npy'), normal_results)

