# Quick training progress checker
# Run this in a NEW terminal: .\check_training_progress.ps1

Write-Host "=== RL Training Progress ===" -ForegroundColor Cyan
Write-Host ""

# Check if process running
$process = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.StartTime -gt (Get-Date).AddHours(-2) }
if ($process) {
    $runtime = (Get-Date) - $process.StartTime
    Write-Host "✅ Training RUNNING" -ForegroundColor Green
    Write-Host "   Process ID: $($process.Id)"
    Write-Host "   Runtime: $($runtime.Hours)h $($runtime.Minutes)m $($runtime.Seconds)s"
}
else {
    Write-Host "❌ No training process found" -ForegroundColor Red
}

Write-Host ""

# Check checkpoints
$checkpoints = Get-ChildItem "runs\rl_training_adaptive\checkpoint_*.pth" -ErrorAction SilentlyContinue
if ($checkpoints) {
    Write-Host "📁 Checkpoints saved:" -ForegroundColor Yellow
    foreach ($cp in $checkpoints | Sort-Object LastWriteTime -Descending) {
        $episode = $cp.Name -replace 'checkpoint_ep|\.pth', ''
        $time = $cp.LastWriteTime.ToString("HH:mm:ss")
        Write-Host "   Episode $episode - $time"
    }
}
else {
    Write-Host "⏳ No checkpoints yet (saves every 50 episodes)" -ForegroundColor Yellow
}

Write-Host ""

# Check if complete
if (Test-Path "runs\rl_training_adaptive\training_log_adaptive.json") {
    Write-Host "🎉 TRAINING COMPLETE!" -ForegroundColor Green
    $log = Get-Content "runs\rl_training_adaptive\training_log_adaptive.json" | ConvertFrom-Json
    Write-Host "   Episodes: $($log.num_episodes)"
    Write-Host "   Time: $([math]::Round($log.training_time/3600, 2)) hours"
    Write-Host "   Avg Reward: $([math]::Round($log.avg_reward, 3))"
    Write-Host "   Avg B: $([math]::Round($log.avg_B, 2))"
}
else {
    Write-Host "⏳ Training in progress..." -ForegroundColor Yellow
    Write-Host "   Expected: ~3-4 hours for 500 episodes"
    Write-Host "   Checkpoints save every 50 episodes"
}
