@echo off
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;F:\project\beifen\DiffMark-main
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
cd /d F:\project\beifen\DiffMark-main
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\probe_mosaic_windows.py --model_path F:\project\beifen\DiffMark-main\results\2026-10-30-20-26-32-353028\model030000.pt --data_dir CelebA-HQ/test_128_500 --codebook_json results/codebook_100x30_min11.json
