@echo off
call "%~dp0run_inswapper128_collusion_mean_100.bat"
if errorlevel 1 exit /b 1
call "%~dp0run_inswapper128_collusion_median_100.bat"
if errorlevel 1 exit /b 1
call "%~dp0run_inswapper128_collusion_mosaic_100.bat"
