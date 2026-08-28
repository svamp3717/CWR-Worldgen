@echo off
setlocal
cd /d "%~dp0"
rem Always inspect with this checkout's source tree, not a stale installed package.
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

if "%~1"=="" goto :usage

set "INPUT=%~f1"
set "ROADS="
if not "%~2"=="" set "ROADS=%~f2"

rem Frozen source bundles normally keep normalized roads beside the generated
rem world. Auto-discover the common layouts, but an explicit second argument wins.
if not defined ROADS if exist "%~dp1normalized\roads.geojson" set "ROADS=%~dp1normalized\roads.geojson"
if not defined ROADS if exist "%~dp1..\normalized\roads.geojson" set "ROADS=%~dp1..\normalized\roads.geojson"

set "OUT=%~dpn1-road-inspector"
if defined ROADS (
  echo Inspecting %INPUT%
  echo Source roads: %ROADS%
  py -m cwr_worldgen.road_inspector_entry "%INPUT%" --roads "%ROADS%" --output "%OUT%"
) else (
  echo Inspecting %INPUT%
  echo No normalized roads.geojson found. Seam checks will still run; source-intersection checks will be skipped.
  py -m cwr_worldgen.road_inspector_entry "%INPUT%" --output "%OUT%"
)
if errorlevel 1 goto :failed

if exist "%OUT%\report.html" start "" "%OUT%\report.html"
echo.
echo Road Inspector report: %OUT%\report.html
exit /b 0

:usage
echo Usage:
echo   INSPECT-ROADS.cmd world.wrp-or-pbo [normalized\roads.geojson]
echo.
echo You can also drag a generated WRP or PBO onto this file.
exit /b 2

:failed
echo.
echo Road Inspector failed.
pause
exit /b 1
