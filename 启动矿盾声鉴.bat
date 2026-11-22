@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m kdsj gui
) else (
  echo 没有找到 .venv 环境，请先按 README 安装。
  pause
)
