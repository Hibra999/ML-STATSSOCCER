@echo off
REM Navigate to the directory of the batch file
cd /d "%~dp0"

REM (Optional) Activate your virtual environment first:
REM call "%USERPROFILE%\python\envs\myenv\Scripts\activate.bat"

REM Run the local web application and pass any provided arguments.
python app.py %*
