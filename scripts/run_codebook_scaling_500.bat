@echo off
set ROOT=F:\project\beifen\DiffMark-main
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;%ROOT%
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
cd /d "%ROOT%"
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\analyze_codebook_scaling.py --input_json results\user_fingerprint_eval_30k_500users_bch_vqgan_probs.json --candidate_sizes 50,100,200,300,500 --repeats 20 --output_json results\codebook_scaling_500users.json --output_md docs\codebook_scaling_500users.md --output_png docs\codebook_scaling_500users.png
