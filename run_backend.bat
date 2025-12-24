@echo off
echo Starting Photobooth Backend...
echo Ensure you have Python installed and added to PATH.

:: Install dependencies if needed (optional, can comment out after first run)
pip install -r requirements.txt

:: Run the server
:: Reload is on for dev, port 10000 as configured
uvicorn main:app --host 0.0.0.0 --port 10000 --reload

pause
