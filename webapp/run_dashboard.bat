@echo off
REM Double-click to open the Cosmic Events dashboard in your browser. Close this window to stop it.
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python -m streamlit run app.py
pause
