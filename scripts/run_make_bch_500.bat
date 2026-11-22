@echo off
set ROOT=F:\project\beifen\DiffMark-main
set PYTHONHOME=D:\Anaconda3\envs\pt2
set PYTHONPATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\shimlib;%ROOT%
set PATH=C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim;C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\DLLs;D:\Anaconda3\envs\pt2;D:\Anaconda3\envs\pt2\Library\bin;C:\WINDOWS\system32;C:\WINDOWS
cd /d "%ROOT%"
C:\Users\WUDONG~1\AppData\Local\Temp\pt2_shim\python.exe scripts\make_bch_codebook.py --num_users 500 --message_length 30 --output_json results\codebook_500x30_bch_min11.json
