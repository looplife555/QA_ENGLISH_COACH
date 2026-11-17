@echo off
title SpeakUp - English Coach
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
  echo Run install_windows.bat first.
  pause
  exit /b 1
)
if exist coach_config.bat call coach_config.bat
call venv\Scripts\activate.bat
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
python -c "import fastapi, uvicorn, multipart" >nul 2>&1 || (
  echo Installing web components, one time only...
  pip install fastapi "uvicorn[standard]" python-multipart
)
set "NV=%~dp0venv\Lib\site-packages\nvidia"
set "PATH=%NV%\cublas\bin;%NV%\cudnn\bin;%LOCALAPPDATA%\Programs\Ollama;%PATH%"
curl -s http://localhost:11434 >nul 2>&1 || (start "" /min ollama serve & timeout /t 5 >nul)
echo Starting SpeakUp... the first start downloads voice models and can take a few minutes.
python server.py
pause
