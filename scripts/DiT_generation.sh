PYTHONPATH=. python eval/DiT_exp2_generation.py --config ./train/config/taskgr13_DiT_exp2.yaml \
    --ckpt_path DiT_exp2_logs/DiT_exp2_taskgr13/version_0/checkpoints/epoch=0-step=6250.ckpt \
    --task hed --shots 5 --batch_size 8 --compute_fid \
    # --task_ckpt_path DiT_tuning_logs/DiT_taskgr13_exp1.2_epoch=0_depth_30_fullft_all_prj/version_0/checkpoints/epoch=33-step=100.ckpt \
