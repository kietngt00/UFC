import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
import os

from annotator.hed import HEDdetector
from annotator.midas import MidasDetector
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.sketch import SketchDetector
from annotator.openpose import OpenposeDetector, util


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--i_start", type=int, default=0, help="Start")
parser.add_argument("--i_end", type=int, default=10, help="End")
args = parser.parse_args()


path = args.path
i_start = args.i_start
i_end = args.i_end

dirs = glob(path + "/*/")

i_end = min(len(dirs), i_end)

hed_annotator = HEDdetector()
midas_annotator = MidasDetector()
canny_annotator = CannyDetector()
mlsd_annotator = MLSDdetector()
sketch_annotator = SketchDetector()
openpose_annotator = OpenposeDetector()


for i in tqdm(range(i_start, i_end)):
    paths = glob(dirs[i] + "/*.jpg")
    for task in ['hed', 'canny', 'mlsd', 'sketch', 'depth', 'normal', 'pose']:
        os.makedirs(dirs[i] + f"/{task}", exist_ok=True)

    for file in paths:
        image = Image.open(file).convert("RGB")
        image = np.array(image)

        name = file.split('/')[-1]

        image_hed = hed_annotator(image)
        Image.fromarray(image_hed).save(dirs[i] + f"/hed/{name}")

        image_depth, image_normal = midas_annotator(image)
        Image.fromarray(image_depth).save(dirs[i] + f"/depth/{name}")
        Image.fromarray(image_normal).save(dirs[i] + f"/normal/{name}" )

        image_canny = canny_annotator(image, 100, 200)
        Image.fromarray(image_canny).save(dirs[i] + f"/canny/{name}")

        image_mlsd = mlsd_annotator(image, 0.1, 0.1)
        Image.fromarray(image_mlsd).save(dirs[i] + f"/mlsd/{name}")

        image_sketch = sketch_annotator(image)
        Image.fromarray(image_sketch).save(dirs[i] + f"/sketch/{name}")
        
        image_pose, _ = openpose_annotator(image)
        Image.fromarray(image_pose).save(dirs[i] + f"/pose/{name}")