@echo off
setlocal
title English Coach - Windows Installer
cd /d "%~dp0"

echo.
echo [1/5] Checking Python (3.10 - 3.12 required)...
set "PY="
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V --version >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  echo Python not found. Installing Python 3.11 via winget...
  winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
  echo.
  echo Python installed. Please CLOSE this window and run install_windows.bat again.
  pause
  exit /b 1
)

echo.
echo [2/5] Creating virtual environment...
if not exist venv %PY% -m venv venv
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [3/5] Installing packages (this can take several minutes)...
nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo No NVIDIA GPU detected - using CPU mode ^(slow for a 27B model^).
  set "DEVICE=cpu"
  pip install torch
) else (
  echo NVIDIA GPU detected - installing CUDA build.
  set "DEVICE=cuda"
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
)
pip install faster-whisper kokoro "misaki[en]" soundfile sounddevice numpy requests fastapi "uvicorn[standard]" python-multipart
if errorlevel 1 (
  echo Package install failed. Check the error above.
  pause
  exit /b 1
)

echo.
echo [4/5] Checking Ollama...
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
where ollama >nul 2>&1
if errorlevel 1 (
  echo Installing Ollama via winget...
  winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
  set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Ollama"
)

echo.
echo [5/5] Pulling the Qwen model.
set "MODEL=qwen3:4b"
set /p MODEL=Enter your Qwen Ollama tag [default: qwen3:4b]: 
curl -s http://localhost:11434 >nul 2>&1 || start "" /min ollama serve
timeout /t 5 >nul
ollama pull %MODEL%

> coach_config.bat echo set "COACH_MODEL=%MODEL%"
>> coach_config.bat echo set "COACH_DEVICE=%DEVICE%"

echo.
echo ===== Done! Double-click run_web.bat to start practicing. =====
pause
