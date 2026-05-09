@echo off
setlocal enabledelayedexpansion

call C:\Xilinx\Vivado\2024.1\settings64.bat

set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%\..\..") do set PROJ_ROOT=%%~fI

set RTL_DIR=%PROJ_ROOT%\hardware\hdl\rtl
set TB_DIR=%PROJ_ROOT%\hardware\hdl\tb
set WORK_DIR=%PROJ_ROOT%\hardware\sim_work
set INC_DIR=%PROJ_ROOT%\config\generated

if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"
cd /d "%WORK_DIR%"

set PASS_TOTAL=0
set FAIL_TOTAL=0

call :run_test core_group "%TB_DIR%\tb_core_group.v" "%RTL_DIR%\core\core_group.v"
call :run_test router_ct "%TB_DIR%\tb_router_ct.v" "%RTL_DIR%\core\event_router_ng.v" "%RTL_DIR%\core\synaptic_connectivity_table.v"
call :run_test integration "%TB_DIR%\tb_integration.v" "%RTL_DIR%\core\core_group.v" "%RTL_DIR%\core\event_router_ng.v" "%RTL_DIR%\core\synaptic_connectivity_table.v"

echo.
echo ============================================================
echo OVERALL RESULTS: %PASS_TOTAL% PASS, %FAIL_TOTAL% FAIL
echo ============================================================

if %FAIL_TOTAL% GTR 0 (
    echo *** SOME TESTS FAILED ***
    exit /b 1
) else (
    echo *** ALL TESTS PASSED ***
    exit /b 0
)

:run_test
set TESTNAME=%~1
set TB_FILE=%~2
shift
shift

set SRC_FILES=
:collect_files
if "%~1"=="" goto files_done
set SRC_FILES=!SRC_FILES! "%~1"
shift
goto collect_files

:files_done
echo.
echo ============================================================
echo Running: %TESTNAME%
echo ============================================================

rmdir /s /q xsim.dir 2>nul
rmdir /s /q .Xil 2>nul
del /q *.log *.jou *.pb *.wdb webtalk* 2>nul

echo [1/3] Compiling...
xvlog -nolog -i "%INC_DIR%" !SRC_FILES! "%TB_FILE%" > compile_%TESTNAME%.log 2>&1
if errorlevel 1 (
    echo [COMPILE ERROR] %TESTNAME%
    type compile_%TESTNAME%.log
    set /a FAIL_TOTAL+=1
    goto :eof
)

for %%F in ("%TB_FILE%") do set TOP=%%~nF

echo [2/3] Elaborating...
xelab -nolog -debug off !TOP! -s %TESTNAME%_sim > elab_%TESTNAME%.log 2>&1
if errorlevel 1 (
    echo [ELABORATE ERROR] %TESTNAME%
    type elab_%TESTNAME%.log
    set /a FAIL_TOTAL+=1
    goto :eof
)

echo [3/3] Simulating...
xsim -nolog %TESTNAME%_sim -runall > sim_%TESTNAME%.log 2>&1
if errorlevel 1 (
    echo [SIM ERROR] %TESTNAME%
    type sim_%TESTNAME%.log
    set /a FAIL_TOTAL+=1
    goto :eof
)

findstr /C:"[PASS]" /C:"[FAIL]" /C:"Results:" /C:"ALL TESTS" /C:"SOME TESTS" /C:"ERROR" sim_%TESTNAME%.log

for /f %%C in ('findstr /C:"[PASS]" sim_%TESTNAME%.log ^| find /C /V ""') do set P=%%C
for /f %%C in ('findstr /C:"[FAIL]" sim_%TESTNAME%.log ^| find /C /V ""') do set F=%%C

set /a PASS_TOTAL+=P
set /a FAIL_TOTAL+=F

goto :eof