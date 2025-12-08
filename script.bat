@echo off
echo ===== Workout Posture Analyzer - Auto Setup =====

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt


echo Running Workout Analyzer...
python src/recorded.py

pause
