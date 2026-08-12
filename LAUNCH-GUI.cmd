@echo off
setlocal
cd /d "%~dp0"
rem Force this archive's src tree ahead of any previously installed cwr-worldgen package.
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
py -m cwr_worldgen.gui
if errorlevel 1 pause
endlocal
