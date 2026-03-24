@echo off
setlocal
echo ==========================================
echo    EEG Schizophrenia Detection System
echo ==========================================
echo.

:: Change to the directory where the script is located
cd /d "%~dp0"
echo Current Directory: %CD%

:: Check for virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment .venv...
    call ".venv\Scripts\activate.bat"
) else (
    echo [WARNING] Virtual environment .venv\Scripts\activate.bat not found.
    echo [INFO] Looking for global Python...
)

:: Verify python version
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python.
    pause
    exit /b
)

:: Check if Streamlit is installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Streamlit not found. Installing dependencies...
    pip install streamlit pandas numpy scikit-learn matplotlib joblib
)

:: Run the app using the explicit streamlit command
echo.
echo [SUCCESS] Launching Streamlit dashboard...
echo.
python -m streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start Streamlit.
    echo Please ensure all dependencies are installed.
    pause
)

endlocal
pause
