# start_bot.ps1
# One-click launcher (and optional watchdog) for the trading bot + uploader.
#
# Usage:
#   .\start_bot.ps1            -- one-shot launch (start bot+uploader if not already running, then exit)
#   .\start_bot.ps1 -Watchdog  -- stay running, restart bot or uploader if either dies
#   .\start_bot.ps1 -Stop      -- stop both processes cleanly
#
# Easier: double-click start_bot.bat / start_watchdog.bat / stop_bot.bat
#
# Prereq: MT5 terminal must already be open and logged in to the Deriv DEMO
# account (the bot can't auto-launch MT5). If MT5 is closed, this script will
# refuse to start the bot.
#
# To run automatically when you log in:
#   Task Scheduler -> Create Basic Task -> Trigger: At log on -> Action: Start a
#   program -> powershell.exe -> Args:
#   -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Xuri\Downloads\trading_bot\start_bot.ps1" -Watchdog

param(
    [switch]$Watchdog,
    [switch]$Stop
)

$ErrorActionPreference = "Continue"
$ProjectRoot = "C:\Users\Educe876\Downloads\Export_trading_bot_PC\trading_bot"
$VenvPython  = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$LogDir      = Join-Path $ProjectRoot "paper_trading\logs"
$BotScript   = Join-Path $ProjectRoot "paper_trading\paper_trader.py"
$UpScript    = Join-Path $ProjectRoot "paper_trading\dashboard_uploader.py"

function Write-Info($msg) { Write-Host "[startup] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[startup] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[startup] $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "[startup] $msg" -ForegroundColor Red }


function Get-BotPids {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match "paper_trader.py" } |
        Select-Object -ExpandProperty ProcessId
}

function Get-UploaderPids {
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match "dashboard_uploader.py" } |
        Select-Object -ExpandProperty ProcessId
}

function Test-Mt5Running {
    # MT5 terminal is "terminal64.exe". Bot can't connect without it.
    $null -ne (Get-Process terminal64 -ErrorAction SilentlyContinue)
}

function Start-Bot {
    if (Get-BotPids) { Write-Info "bot already running"; return }
    if (-not (Test-Mt5Running)) {
        Write-Err "MT5 terminal is NOT running. Start MT5 and log in, then re-run this script."
        return
    }
    Write-Info "starting bot..."
    $env:PYTHONIOENCODING = "utf-8"
    $stdout = Join-Path $LogDir "paper_trader_stdout.log"
    $stderr = Join-Path $LogDir "paper_trader_stderr.log"
    Start-Process -FilePath $VenvPython `
        -ArgumentList "-u", $BotScript `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 2
    if (Get-BotPids) {
        Write-Ok "bot launched (PID $((Get-BotPids) -join ', '))"
    } else {
        Write-Err "bot failed to launch -- check $stderr"
    }
}

function Start-Uploader {
    if (Get-UploaderPids) { Write-Info "uploader already running"; return }
    Write-Info "starting uploader..."
    $env:PYTHONIOENCODING = "utf-8"
    $stdout = Join-Path $LogDir "uploader_stdout.log"
    $stderr = Join-Path $LogDir "uploader_stderr.log"
    Start-Process -FilePath $VenvPython `
        -ArgumentList "-u", $UpScript `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 2
    if (Get-UploaderPids) {
        Write-Ok "uploader launched (PID $((Get-UploaderPids) -join ', '))"
    } else {
        Write-Err "uploader failed to launch -- check $stderr"
    }
}

function Stop-Bot {
    $pids = Get-BotPids
    if (-not $pids) { Write-Info "bot not running"; return }
    Write-Info "stopping bot (PID $($pids -join ', '))"
    $pids | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    Write-Ok "bot stopped"
}

function Stop-Uploader {
    $pids = Get-UploaderPids
    if (-not $pids) { Write-Info "uploader not running"; return }
    Write-Info "stopping uploader (PID $($pids -join ', '))"
    $pids | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
    Write-Ok "uploader stopped"
}

# Stop mode
if ($Stop) {
    Stop-Bot
    Stop-Uploader
    exit 0
}

# Initial launch
Write-Info "project: $ProjectRoot"
Write-Info "MT5 terminal running: $(Test-Mt5Running)"
Start-Bot
Start-Uploader

# Watchdog mode
if (-not $Watchdog) {
    Write-Ok "one-shot launch complete. Re-run with -Watchdog to auto-restart on crashes."
    exit 0
}

Write-Info "entering watchdog mode (60s checks). Ctrl+C to stop the watchdog (does NOT stop bot/uploader)."
$lastReport = Get-Date
while ($true) {
    Start-Sleep -Seconds 60

    if (-not (Get-BotPids)) {
        Write-Warn "bot is down -- restarting"
        Start-Bot
    }
    if (-not (Get-UploaderPids)) {
        Write-Warn "uploader is down -- restarting"
        Start-Uploader
    }

    if ((Get-Date) - $lastReport -gt (New-TimeSpan -Minutes 30)) {
        $bot = Get-BotPids
        $up  = Get-UploaderPids
        $mt5 = Test-Mt5Running
        Write-Info "heartbeat -- bot=[$($bot -join ',')] uploader=[$($up -join ',')] mt5=$mt5"
        $lastReport = Get-Date
    }
}
