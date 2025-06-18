img_dir=$1 # "path/*.png"
out_dir=$2 # "path/result.pt"


cd /path/to/detectron2/projects/DensePose

python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \
https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
"$img_dir" --output $out_dir -v

cd /path/to/ufc