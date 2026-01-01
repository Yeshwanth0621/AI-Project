# Quick Start Script for SLM Training System
# Run this script to set up and train your model

Write-Host "=" -NoNewline; 1..60 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host "Activity-Field Mapping SLM - Quick Start"
Write-Host "=" -NoNewline; 1..60 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host ""

# Navigate to ml directory
Set-Location -Path "e:\Git\Test\ml"

Write-Host "Step 1: Installing dependencies..."
Write-Host "   This may take a few minutes..."
Write-Host ""
pip install -r requirements.txt --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies. Please run manually:" -ForegroundColor Red
    Write-Host "   pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

Write-Host "SUCCESS: Dependencies installed!" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Preparing data..."
python prepare_data.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to prepare data." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 3: Training model..."
Write-Host "   This will take 5-15 minutes on CPU"
Write-Host "   You can monitor the training progress below:"
Write-Host ""

python train_model.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to train model." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Step 4: Testing model..."
python inference.py --test

Write-Host ""
Write-Host "=" -NoNewline; 1..60 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=" -NoNewline; 1..60 | ForEach-Object { Write-Host "=" -NoNewline }; Write-Host ""
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Start API server: python backend\ml_service.py"
Write-Host "  2. Open frontend: frontend\activity_predictor.html"
Write-Host "  3. Try interactive mode: python inference.py --interactive"
Write-Host ""
Write-Host "For more details, see README.md"
Write-Host ""
