@echo off
set ROOT=F:\project\beifen\DiffMark-main
set OPENAI_LOGDIR=%ROOT%\logs\user_fingerprint_500users_bch_vqgan
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;%ROOT%
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
if not exist "%OPENAI_LOGDIR%" mkdir "%OPENAI_LOGDIR%"
cd /d "%ROOT%"
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\user_fingerprint_eval.py --image_size 128 --message_length 30 --embedding_dim 256 --num_channels 32 --num_heads 4 --num_res_blocks 1 --attention_resolutions 16,8 --diffusion_steps 100 --timestep_respacing ddim100 --model_path results\2026-10-30-20-26-32-353028\model030000.pt --data_dir CelebA-HQ\test_128_500 --batch_size 2 --num_users 500 --images_per_user 1 --codebook_json results\codebook_500x30_bch_min11.json --noise_layer "VQGAN()" --output_json results\user_fingerprint_eval_30k_500users_bch_vqgan_probs.json
