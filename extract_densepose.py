import os
import glob
import subprocess

# Path to the root directory containing subfolders
root_dir = "/path/to/laion_human"

os.chdir('/path/to/detectron2/projects/DensePose')

# Set model config and weights
config_file = "configs/densepose_rcnn_R_50_FPN_s1x.yaml"
model_weights = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"

# Loop through all subdirectories in the root
for subdir in sorted(os.listdir(root_dir))[1:]:
    sub_path = os.path.join(root_dir, subdir)
    if os.path.isdir(sub_path):
        input_pattern = os.path.join(sub_path, "*.jpg")
        output_dir = os.path.join(sub_path, "densepose")

        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Build the command
        command = [
            "CUDA_VISIBLE_DEVICES=2", "python", "apply_net.py", "show",
            config_file,
            model_weights,
            f'"{input_pattern}"',
            "--output", f'"{os.path.join(output_dir, "*.jpg")}"',
            "dp_segm", "-v"
        ]

        # Print the command (optional for debugging)
        print("Running command for:", sub_path)
        print(" ".join(command))

        # Run the command
        subprocess.run(" ".join(command), shell=True)
