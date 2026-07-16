@echo off
REM Double-click to start bot + uploader AND keep monitoring.
REM Restarts either if it dies. Closes when you close this window.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_bot.ps1" -Watchdog
