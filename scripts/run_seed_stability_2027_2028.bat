@echo off
setlocal enabledelayedexpansion
set ROOT=F:\project\beifen\DiffMark-main
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;%ROOT%
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
cd /d "%ROOT%"
set PY=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe
set MODEL=results\2026-10-30-20-26-32-353028\model030000.pt
set CODEBOOK=results\codebook_100x30_min11.json
set BASE=--image_size 128 --message_length 30 --embedding_dim 256 --num_channels 32 --num_heads 4 --num_res_blocks 1 --attention_resolutions 16,8 --diffusion_steps 100 --timestep_respacing ddim100 --model_path %MODEL% --data_dir CelebA-HQ\test_128_500 --num_users 100 --codebook_json %CODEBOOK% --code_seed 2026

for %%S in (2027 2028) do (
  set SEED=%%S
  set OUTDIR=results\seed_stability\seed%%S
  if not exist "!OUTDIR!" mkdir "!OUTDIR!"

  set OPENAI_LOGDIR=%ROOT%\logs\seed_stability\seed%%S_fingerprint_vqgan
  if not exist "!OPENAI_LOGDIR!" mkdir "!OPENAI_LOGDIR!"
  echo [%date% %time%] START seed=%%S fingerprint vqgan
  %PY% scripts\user_fingerprint_eval.py %BASE% --batch_size 4 --images_per_user 5 --seed %%S --noise_layer "VQGAN()" --output_json !OUTDIR!\fingerprint_vqgan.json
  if errorlevel 1 echo [%date% %time%] FAILED seed=%%S fingerprint vqgan

  set OPENAI_LOGDIR=%ROOT%\logs\seed_stability\seed%%S_fingerprint_simswap224
  if not exist "!OPENAI_LOGDIR!" mkdir "!OPENAI_LOGDIR!"
  echo [%date% %time%] START seed=%%S fingerprint simswap224
  %PY% scripts\user_fingerprint_eval.py %BASE% --batch_size 4 --images_per_user 5 --seed %%S --noise_layer "SimSwap(224)" --output_json !OUTDIR!\fingerprint_simswap224.json
  if errorlevel 1 echo [%date% %time%] FAILED seed=%%S fingerprint simswap224

  set OPENAI_LOGDIR=%ROOT%\logs\seed_stability\seed%%S_fingerprint_inswapper128
  if not exist "!OPENAI_LOGDIR!" mkdir "!OPENAI_LOGDIR!"
  echo [%date% %time%] START seed=%%S fingerprint inswapper128
  %PY% scripts\user_fingerprint_eval.py %BASE% --batch_size 4 --images_per_user 5 --seed %%S --noise_layer "InSwapper(128)" --output_json !OUTDIR!\fingerprint_inswapper128.json
  if errorlevel 1 echo [%date% %time%] FAILED seed=%%S fingerprint inswapper128

  set OPENAI_LOGDIR=%ROOT%\logs\seed_stability\seed%%S_collusion_vqgan_mean
  if not exist "!OPENAI_LOGDIR!" mkdir "!OPENAI_LOGDIR!"
  echo [%date% %time%] START seed=%%S collusion vqgan mean
  %PY% scripts\user_collusion_eval.py %BASE% --batch_size 2 --num_groups 100 --min_k 2 --max_k 5 --mix mean --noise_layer "VQGAN()" --identity_offset 0 --seed %%S --collusion_seed 2026 --output_json !OUTDIR!\collusion_vqgan_mean.json
  if errorlevel 1 echo [%date% %time%] FAILED seed=%%S collusion vqgan mean

  set OPENAI_LOGDIR=%ROOT%\logs\seed_stability\seed%%S_collusion_simswap224_mean
  if not exist "!OPENAI_LOGDIR!" mkdir "!OPENAI_LOGDIR!"
  echo [%date% %time%] START seed=%%S collusion simswap224 mean
  %PY% scripts\user_collusion_eval.py %BASE% --batch_size 2 --num_groups 100 --min_k 2 --max_k 5 --mix mean --noise_layer "SimSwap(224)" --identity_offset 1 --seed %%S --collusion_seed 2026 --output_json !OUTDIR!\collusion_simswap224_mean.json
  if errorlevel 1 echo [%date% %time%] FAILED seed=%%S collusion simswap224 mean

  set OPENAI_LOGDIR=%ROOT%\logs\seed_stability\seed%%S_collusion_simswap224_median
  if not exist "!OPENAI_LOGDIR!" mkdir "!OPENAI_LOGDIR!"
  echo [%date% %time%] START seed=%%S collusion simswap224 median
  %PY% scripts\user_collusion_eval.py %BASE% --batch_size 2 --num_groups 100 --min_k 2 --max_k 5 --mix median --noise_layer "SimSwap(224)" --identity_offset 1 --seed %%S --collusion_seed 2026 --output_json !OUTDIR!\collusion_simswap224_median.json
  if errorlevel 1 echo [%date% %time%] FAILED seed=%%S collusion simswap224 median

  set OPENAI_LOGDIR=%ROOT%\logs\seed_stability\seed%%S_collusion_simswap224_mosaic
  if not exist "!OPENAI_LOGDIR!" mkdir "!OPENAI_LOGDIR!"
  echo [%date% %time%] START seed=%%S collusion simswap224 mosaic
  %PY% scripts\user_collusion_eval.py %BASE% --batch_size 2 --num_groups 100 --min_k 2 --max_k 5 --mix mosaic --noise_layer "SimSwap(224)" --identity_offset 1 --seed %%S --collusion_seed 2026 --output_json !OUTDIR!\collusion_simswap224_mosaic.json
  if errorlevel 1 echo [%date% %time%] FAILED seed=%%S collusion simswap224 mosaic
)
echo [%date% %time%] ALL DONE