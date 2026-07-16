@echo off
REM Stop both bot and uploader cleanly.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_bot.ps1" -Stop
pause
