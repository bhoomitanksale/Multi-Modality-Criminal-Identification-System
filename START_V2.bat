@echo off
title CIS v2 - Criminal Identification System
color 0B
cls
echo.
echo  =========================================================================
echo   CIS v2  Criminal Identification System  BioFuse  IGNISIA
echo  =========================================================================
echo.
cd /d "%~dp0"

echo  [1/6] Checking Python...
python --version >nul 2>nul
if errorlevel 1 (
    color 0C
    echo  [ERROR] Python not found.
    echo         Install from: https://www.python.org/downloads/
    echo         Make sure to tick: Add Python to PATH
    echo.
    pause
    exit /b 1
)
echo  [OK]  Python found

echo.
echo  [2/6] Checking packages...
python -c "import cv2, PIL, numpy" >nul 2>nul
if errorlevel 1 (
    echo  [*]  Installing packages, please wait...
    pip install opencv-contrib-python pillow numpy --quiet
    if errorlevel 1 (
        color 0C
        echo  [ERROR] Could not install packages.
        echo         Run manually: pip install opencv-contrib-python pillow numpy
        pause
        exit /b 1
    )
    echo  [OK]  Packages installed
) else (
    echo  [OK]  Packages ready
    python -c "import cv2; cv2.face.LBPHFaceRecognizer_create()" >nul 2>nul
    if errorlevel 1 pip install opencv-contrib-python --quiet >nul 2>nul
)

echo.
echo  [3/6] mediapipe check...
python -c "import mediapipe" >nul 2>nul
if errorlevel 1 (
    echo  [INFO] mediapipe not found - gait uses fallback (OK)
) else (
    echo  [OK]  mediapipe available
)

echo.
echo  [4/6] Camera probe...
python -c "import cv2;c=cv2.VideoCapture(0);c.release()" >nul 2>nul
echo  [OK]  Camera probed

echo.
echo  [5/6] Criminal reference folders...
if not exist data mkdir data
if not exist data\criminal_faces mkdir data\criminal_faces
if not exist data\criminal_faces\dawood_ibrahim mkdir data\criminal_faces\dawood_ibrahim
if not exist data\criminal_faces\osama_bin_laden mkdir data\criminal_faces\osama_bin_laden
if not exist data\criminal_faces\el_chapo mkdir data\criminal_faces\el_chapo
if not exist data\criminal_faces\pablo_escobar mkdir data\criminal_faces\pablo_escobar
if not exist data\criminal_faces\hafiz_saeed mkdir data\criminal_faces\hafiz_saeed
if not exist data\criminal_faces\ted_bundy mkdir data\criminal_faces\ted_bundy
echo  [OK]  Folders ready

echo.
echo  [6/6] Initializing database...
python -c "from database.db_manager import init_db;init_db()" >nul 2>nul
echo  [OK]  Database ready

echo.
echo  =========================================================================
echo   Tab 1 - Image Upload  : Photo + HIGH PRIORITY criminal match
echo   Tab 2 - Video Upload  : Frame-by-frame CCTV
echo   Tab 3 - Live Webcam   : 3-scan pipeline + Add to DB
echo   Tab 4 - Database      : Criminal registry + session log
echo  =========================================================================
echo.
echo  Launching CIS v2...
echo.

python cis_v2.py

echo.
if errorlevel 1 (
    color 0C
    echo  [ERROR] CIS v2 crashed.
    echo  Fix: pip install --upgrade opencv-contrib-python pillow numpy
    echo.
) else (
    color 0A
    echo  [OK] CIS v2 closed normally.
)
echo.
pause
