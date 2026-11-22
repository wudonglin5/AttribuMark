@echo off
call "%~dp0run_inswapper128_collusion_mean_10_exploratory.bat"
if errorlevel 1 exit /b 1
call "%~dp0run_inswapper128_collusion_median_10_exploratory.bat"
if errorlevel 1 exit /b 1
call "%~dp0run_inswapper128_collusion_mosaic_10_exploratory.bat"
