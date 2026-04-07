@echo off
echo ============================================================
echo    Patient No-Show Prediction - Full Pipeline Runner
echo    ML Design Final Project
echo ============================================================
echo.

:: Step 1: Install dependencies
echo [1/5] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo      Done.
echo.

:: Step 2: Run ETL Pipeline
echo [2/5] Running ETL Pipeline (Bronze → Silver → Gold)...
python Data_Ingestion.py
if %errorlevel% neq 0 (
    echo ERROR: ETL Pipeline failed.
    pause
    exit /b 1
)
echo      Done.
echo.

:: Step 3: Train Models
echo [3/5] Training Models + MLflow Logging...
python model_training.py
if %errorlevel% neq 0 (
    echo ERROR: Model training failed.
    pause
    exit /b 1
)
echo      Done.
echo.

:: Step 4: Generate Predictions
echo [4/5] Generating Predictions + Top 30 High-Risk List...
python predict.py
if %errorlevel% neq 0 (
    echo ERROR: Prediction failed.
    pause
    exit /b 1
)
echo      Done.
echo.

:: Step 5: Launch Streamlit Dashboard
echo [5/5] Launching Streamlit Dashboard...
echo      Dashboard will open in your browser.
echo      Press Ctrl+C in this window to stop the server.
echo.
streamlit run app.py