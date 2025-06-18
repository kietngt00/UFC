path1=$1
path2=$2
device=$3
python -m pytorch_fid $path1 $path2 --device cuda:$device