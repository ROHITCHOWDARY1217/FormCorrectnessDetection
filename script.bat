@echo off
echo ===== Workout Posture Analyzer - Auto Setup =====

REM 
set PY_CMD=py -3.10

REM 
if not exist venv (
    echo Creating virtual environment with Python 3.10...
    %PY_CMD% -m venv venv
)

echo Activating environment...
call venv\Scripts\activate

echo Python version in venv:
python -V

echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Running Workout Analyzer...
python src\recorded.py

pause