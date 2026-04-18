param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("up", "down", "restart", "logs", "ps", "smoke")]
    [string]$Action,
    [string]$Service = "",
    [switch]$Build
)

$ErrorActionPreference = "Stop"

switch ($Action) {
    "up" {
        if ($Build) {
            docker compose up --build -d
        } else {
            docker compose up -d
        }
        break
    }
    "down" {
        docker compose down
        break
    }
    "restart" {
        docker compose restart
        break
    }
    "logs" {
        if ([string]::IsNullOrWhiteSpace($Service)) {
            docker compose logs -f --tail=200
        } else {
            docker compose logs -f --tail=200 $Service
        }
        break
    }
    "ps" {
        docker compose ps
        break
    }
    "smoke" {
        $runningServices = docker compose ps --services --status running
        $requiredServices = @("api", "worker", "redis")
        $missing = @()
        foreach ($svc in $requiredServices) {
            if ($runningServices -notcontains $svc) {
                $missing += $svc
            }
        }

        if ($missing.Count -gt 0) {
            throw "Required services are not running: $($missing -join ', '). Run './scripts/dev-compose.ps1 -Action up -Build' first."
        }

        & "$PSScriptRoot/smoke-compose.ps1"
        break
    }
}

