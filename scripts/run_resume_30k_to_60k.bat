@echo off
set OPENAI_LOGDIR=F:\project\beifen\DiffMark-main\results\resume_30k_to_60k
set PYTHONPATH=F:\project\beifen\DiffMark-main
set PATH=D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;%PATH%
cd /d F:\project\beifen\DiffMark-main
D:\Anaconda3\envs\pt2\python.exe scripts\image_train.py ^
  --attention_resolutions 16,8 --image_size 128 --message_length 30 ^
  --embedding_dim 256 --num_channels 32 --num_heads 4 --num_res_blocks 1 ^
  --data_dir F:\project\beifen\DiffMark-main\CelebA-HQ\train_128 ^
  --batch_size 8 --lr 1e-4 --lr_anneal_steps 60000 --weight_decay 1e-5 ^
  --threshold 5000 --save_interval 2000 ^
  --resume_checkpoint F:\project\beifen\DiffMark-main\results\2026-10-30-20-26-32-353028\model030000.pt
pause
