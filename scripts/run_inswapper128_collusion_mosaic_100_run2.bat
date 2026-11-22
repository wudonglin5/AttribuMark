@echo off
set ROOT=F:\project\beifen\DiffMark-main
set OPENAI_LOGDIR=%ROOT%\logs\inswapper128_collusion_mosaic_100_run2
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;%ROOT%
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
if not exist "%OPENAI_LOGDIR%" mkdir "%OPENAI_LOGDIR%"
cd /d "%ROOT%"
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\user_collusion_eval.py --image_size 128 --message_length 30 --embedding_dim 256 --num_channels 32 --num_heads 4 --num_res_blocks 1 --attention_resolutions 16,8 --diffusion_steps 100 --timestep_respacing ddim100 --model_path results\2026-10-30-20-26-32-353028\model030000.pt --data_dir CelebA-HQ\test_128_500 --batch_size 2 --num_users 100 --num_groups 100 --min_k 2 --max_k 5 --mix mosaic --codebook_json results\codebook_100x30_min11.json --noise_layer "InSwapper(128)" --identity_offset 1 --output_json results\user_collusion_eval_30k_min11_inswapper128_mosaic_100groups.json
