@echo off
REM Double-click to start bot + uploader (one-shot launch).
REM Wraps start_bot.ps1 with -ExecutionPolicy Bypass.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_bot.ps1"
pause
