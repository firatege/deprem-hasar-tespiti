param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [int]$TimeoutSec = 120,
    [int]$PollIntervalSec = 2
)

$ErrorActionPreference = "Stop"

Write-Host "[smoke] Checking API readiness..."
$readyDeadline = (Get-Date).AddSeconds($TimeoutSec)
$isReady = $false
while ((Get-Date) -lt $readyDeadline) {
    try {
        $ready = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get
        if ($ready.status -eq "ok") {
            $isReady = $true
            break
        }
    }
    catch {
        Write-Host "[smoke] API not ready yet: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $PollIntervalSec
}

if (-not $isReady) {
    throw "API readiness check failed within ${TimeoutSec}s"
}

$payload = @{
    event_id = "usgs-smoke-event"
    tile_id = "tile-smoke-001"
    pga_value = 0.42
    damage_label = "major"
}

Write-Host "[smoke] Submitting async job..."
$submit = Invoke-RestMethod -Uri "$BaseUrl/jobs/submit" -Method Post -ContentType "application/json" -Body ($payload | ConvertTo-Json -Compress)
if (-not $submit.job_id) {
    throw "Submit response does not include job_id"
}

$jobId = [string]$submit.job_id
Write-Host "[smoke] job_id=$jobId"

$deadline = (Get-Date).AddSeconds($TimeoutSec)
$status = "queued"

while ((Get-Date) -lt $deadline) {
    $statusResponse = Invoke-RestMethod -Uri "$BaseUrl/jobs/status/$jobId" -Method Get
    $status = [string]$statusResponse.status
    Write-Host "[smoke] status=$status"

    if ($status -eq "succeeded") {
        break
    }
    if ($status -in @("failed", "canceled")) {
        throw "Job finished in unexpected state: $status"
    }

    Start-Sleep -Seconds $PollIntervalSec
}

if ($status -ne "succeeded") {
    if ($status -eq "queued") {
        throw "Timed out with status=queued within ${TimeoutSec}s. Worker may not be processing jobs. Check: 'docker compose ps', 'docker compose logs --tail=200 worker', and 'docker compose logs --tail=200 redis'."
    }
    throw "Timed out waiting for succeeded status within ${TimeoutSec}s (last status=$status)"
}

Write-Host "[smoke] Fetching result..."
$result = Invoke-RestMethod -Uri "$BaseUrl/jobs/result/$jobId" -Method Get
if ($null -eq $result.binary_damage) {
    throw "Result does not include binary_damage"
}
if ([int]$result.binary_damage -ne 1) {
    throw "Expected binary_damage=1 for major label; got $($result.binary_damage)"
}

Write-Host "[smoke] OK - async flow succeeded" -ForegroundColor Green

