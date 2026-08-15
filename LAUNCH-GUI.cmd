@echo off
setlocal
cd /d "%~dp0"
rem Force this archive's src tree ahead of any previously installed cwr-worldgen package.
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
rem Keep GUI state and generated GUI mapping settings beside build and source-data.
if not exist "%~dp0config" mkdir "%~dp0config"
set "CWR_WORLDGEN_GUI_STATE=%~dp0config\gui-state.json"
py -m cwr_worldgen.gui
if errorlevel 1 pause
endlocal
