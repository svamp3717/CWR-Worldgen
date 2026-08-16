@echo off
setlocal
cd /d "%~dp0\..\.."
python -m pip install -e ".[sources]" || exit /b 1
python "%~dp0build_example.py" %*
