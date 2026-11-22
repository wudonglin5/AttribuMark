@echo off
set ROOT=F:\project\beifen\DiffMark-main
set OPENAI_LOGDIR=%ROOT%\logs\simswap224_attack_strength_100
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;%ROOT%
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
if not exist "%OPENAI_LOGDIR%" mkdir "%OPENAI_LOGDIR%"
cd /d "%ROOT%"
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\eval_simswap_attack_strength.py --image_size 128 --message_length 30 --embedding_dim 256 --num_channels 32 --num_heads 4 --num_res_blocks 1 --attention_resolutions 16,8 --diffusion_steps 100 --timestep_respacing ddim100 --num_images 100 --codebook_json results\codebook_100x30_min11.json --arcface_checkpoint noise_layers\simswap\arcface_checkpoint.tar --output_json results\simswap224_attack_strength.json --output_md docs\simswap224_attack_strength.md --output_png docs\simswap224_attack_strength.png
