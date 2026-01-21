# Create package for Experiments 3 & 4
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Creating Experiment 3 & 4 Package" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Create base directory
$packageDir = "experiment_3_4_package"
Write-Host "Creating package directory..." -ForegroundColor Yellow
Remove-Item -Recurse -Force $packageDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $packageDir | Out-Null

# Copy scripts
Write-Host "Copying scripts..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$packageDir\scripts" | Out-Null
Copy-Item "scripts\evaluate_rl_agent.py" -Destination "$packageDir\scripts\" -Force
Copy-Item "scripts\statistical_tests.py" -Destination "$packageDir\scripts\" -Force

# Copy source code
Write-Host "Copying source code..." -ForegroundColor Yellow
Copy-Item -Recurse -Force "src" "$packageDir\src"

# Copy models
Write-Host "Copying trained models..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$packageDir\runs\rl_training" | Out-Null
Copy-Item "runs\rl_training\best_model.pth" -Destination "$packageDir\runs\rl_training\" -Force

New-Item -ItemType Directory -Force -Path "$packageDir\runs\train\yolo_cure_tsd\weights" | Out-Null
Copy-Item "runs\train\yolo_cure_tsd\weights\best.pt" -Destination "$packageDir\runs\train\yolo_cure_tsd\weights\" -Force

# Copy masks
Write-Host "Copying compression masks..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$packageDir\data\masks" | Out-Null
Copy-Item "data\masks\*.npy" -Destination "$packageDir\data\masks\" -Force

# Copy dataset config
Write-Host "Copying dataset configuration..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$packageDir\data\yolo_dataset_full" | Out-Null
Copy-Item "data\yolo_dataset_full\data.yaml" -Destination "$packageDir\data\yolo_dataset_full\" -Force
Copy-Item "data\yolo_dataset_full\train_val_split.txt" -Destination "$packageDir\data\yolo_dataset_full\" -Force

# Copy validation videos ONLY (280 videos starting with 02_)
Write-Host "Copying validation videos only (280 videos from cure-tsd)..." -ForegroundColor Yellow

# Create directories
New-Item -ItemType Directory -Force -Path "$packageDir\data\cure-tsd\data" | Out-Null
New-Item -ItemType Directory -Force -Path "$packageDir\data\cure-tsd\labels" | Out-Null

# Copy only 02_ videos and their labels
$count = 0
Get-ChildItem "data\cure-tsd\data" -Filter "02_*.mp4" | ForEach-Object {
  Copy-Item $_.FullName -Destination "$packageDir\data\cure-tsd\data\" -Force
  $count++
    
  # Copy corresponding label file
  $labelFile = "data\cure-tsd\labels\$($_.BaseName).txt"
  if (Test-Path $labelFile) {
    Copy-Item $labelFile -Destination "$packageDir\data\cure-tsd\labels\" -Force
  }
}

Write-Host "Copied $count validation videos" -ForegroundColor Cyan

# Copy fixed baseline results
Write-Host "Copying Experiment 1 results..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "$packageDir\outputs" | Out-Null
Copy-Item "outputs\fixed_baseline_results.csv" -Destination "$packageDir\outputs\" -Force

# Copy requirements
Write-Host "Copying requirements..." -ForegroundColor Yellow
Copy-Item "requirements.txt" -Destination "$packageDir\requirements.txt" -Force

# Create README for the package
Write-Host "Creating setup instructions..." -ForegroundColor Yellow
$readme = @"
# Experiment 3 & 4 Package
# ========================

## Setup Instructions

1. Extract this package
2. Create conda environment:
   ``````
   conda create -n rl_video_compression python=3.10
   conda activate rl_video_compression
   ``````

3. Install dependencies:
   ``````
   pip install -r requirements.txt
   ``````

4. Set PYTHONPATH and run experiments:
   ``````powershell
   `$env:PYTHONPATH = "FULL_PATH_TO_THIS_FOLDER"
   python scripts\evaluate_rl_agent.py
   python scripts\statistical_tests.py
   ``````

## Package Contents

- **scripts/**
  - evaluate_rl_agent.py - Experiment 3: Test RL agent
  - statistical_tests.py - Experiment 4: Statistical analysis

- **src/** - Source code (phase1, phase4, phase5)

- **runs/**
  - rl_training/best_model.pth - Trained RL agent
  - train/yolo_cure_tsd/weights/best.pt - YOLO model

- **data/**
  - masks/ - SCI compression masks (B=6,8,10,12,15,20)
  - yolo_dataset_full/ - Validation videos + labels

- **outputs/**
  - fixed_baseline_results.csv - Experiment 1 results for comparison

## Expected Runtime

- Experiment 3: ~3-4 hours (280 videos)
- Experiment 4: <1 minute (statistical tests)

## Expected Outputs

- outputs/rl_agent_results.csv (280 rows)
- outputs/statistical_tests_results.json
- Console output with summary statistics
"@

Set-Content -Path "$packageDir\README.md" -Value $readme

# Create zip
Write-Host "`nCreating ZIP archive..." -ForegroundColor Yellow
$zipPath = "experiment_3_4_package.zip"
Remove-Item $zipPath -ErrorAction SilentlyContinue
Compress-Archive -Path "$packageDir\*" -DestinationPath $zipPath -Force

# Show summary
$zipSize = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Package created successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "File: $zipPath" -ForegroundColor Cyan
Write-Host "Size: $zipSize MB" -ForegroundColor Cyan
Write-Host "`nContents:" -ForegroundColor Yellow
Get-ChildItem -Recurse $packageDir | Measure-Object | Select-Object -ExpandProperty Count | ForEach-Object { Write-Host "   Total files: $_" -ForegroundColor White }
Write-Host "`nReady to transfer to other computer!" -ForegroundColor Green
