@echo off
set ROOT=F:\project\beifen\DiffMark-main
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;%ROOT%
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
cd /d "%ROOT%"
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\analyze_multimage_llr.py --input_json results\user_fingerprint_eval_30k_min11_simswap224_probs.json --output_json results\simswap224_multimage_llr_summary.json --output_txt results\simswap224_multimage_llr_summary.txt
if errorlevel 1 exit /b %errorlevel%
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\analyze_collusion_by_k.py --input_json results\user_collusion_eval_30k_min11_simswap224_mean_100groups.json --codebook_json results\codebook_100x30_min11.json --output_txt results\simswap224_collusion_soft_detector_mean_100groups.txt --output_json results\simswap224_collusion_soft_detector_mean_100groups.json --attack_label vqgan --attack_name SimSwap224
if errorlevel 1 exit /b %errorlevel%
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\compare_attacks.py --vqgan_json results\user_fingerprint_eval_30k_min11_probs.json --simswap_json results\user_fingerprint_eval_30k_min11_simswap224_probs.json --vqgan_llr results\multimage_llr_summary.json --simswap_llr results\simswap224_multimage_llr_summary.json --output_md results\simswap224_comparison.md --output_png docs\simswap224_curve.png
if errorlevel 1 exit /b %errorlevel%
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\plot_simswap_collusion.py --summary_json results\simswap224_collusion_soft_detector_mean_100groups.json --output_png docs\simswap224_collusion_curve.png --attack_name SimSwap224 --vqgan_summary_json results\vqgan_collusion_soft_detector_mean_100groups.json
