@echo off
cd /d "%~dp0"
echo Starting Streamlit app...
echo.
python -m streamlit run Works.py --server.port 8501
pause
