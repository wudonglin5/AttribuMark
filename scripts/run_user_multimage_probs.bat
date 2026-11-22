@echo off
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;F:\project\beifen\DiffMark-main
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
cd /d F:\project\beifen\DiffMark-main
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\user_fingerprint_eval.py --image_size 128 --message_length 30 --embedding_dim 256 --num_channels 32 --num_heads 4 --num_res_blocks 1 --attention_resolutions 16,8 --diffusion_steps 100 --timestep_respacing ddim100 --model_path F:\project\beifen\DiffMark-main\results\2026-10-30-20-26-32-353028\model030000.pt --data_dir CelebA-HQ/test_128_500 --batch_size 4 --num_users 100 --images_per_user 5 --codebook_json results/codebook_100x30_min11.json --output_json results/user_fingerprint_eval_30k_min11_probs.json
