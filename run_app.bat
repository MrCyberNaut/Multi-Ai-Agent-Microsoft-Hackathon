@echo off
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Starting Streamlit app...
python -m streamlit run app.py
echo.
echo If you see errors, try running these commands manually:
echo pip install streamlit
echo python -m streamlit run app.py
pause 