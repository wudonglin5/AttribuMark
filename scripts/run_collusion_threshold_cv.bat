@echo off
set ROOT=F:\project\beifen\DiffMark-main
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;%ROOT%
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
cd /d "%ROOT%"
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\calibrate_collusion_threshold_cv.py --input_json "VQGAN mean=results\user_collusion_eval_30k_min11_mean_probs_100groups.json" --input_json "SimSwap mean=results\user_collusion_eval_30k_min11_simswap224_mean_100groups.json" --input_json "SimSwap median=results\user_collusion_eval_30k_min11_simswap224_median_100groups.json" --input_json "SimSwap mosaic=results\user_collusion_eval_30k_min11_simswap224_mosaic_100groups.json" --output_json results\collusion_threshold_cv.json --output_md docs\collusion_threshold_cv.md --output_png docs\collusion_threshold_cv.png
