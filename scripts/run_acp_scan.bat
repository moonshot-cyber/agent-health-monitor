@echo off
REM ──────────────────────────────────────────────────
REM  ACP Nightly Scan — Windows Task Scheduler runner
REM  Schedule: Daily at 02:00 UTC via Task Scheduler
REM ──────────────────────────────────────────────────

set PROJECT_DIR=C:\Users\Pablo\agent-health-monitor
set LOG_DIR=%PROJECT_DIR%\logs
set TIMESTAMP=%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOG_FILE=%LOG_DIR%\acp_scan_%TIMESTAMP%.log

REM Create logs directory if missing
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%DATE% %TIME%] Starting ACP nightly scan >> "%LOG_FILE%" 2>&1

cd /d "%PROJECT_DIR%"

REM Run the scan with output to both console and log file
python cron_acp_scan.py --max-agents 500 --max-scans 100 >> "%LOG_FILE%" 2>&1

set EXIT_CODE=%ERRORLEVEL%

echo [%DATE% %TIME%] Scan finished with exit code %EXIT_CODE% >> "%LOG_FILE%" 2>&1

REM Clean up logs older than 30 days
forfiles /p "%LOG_DIR%" /m "acp_scan_*.log" /d -30 /c "cmd /c del @path" 2>nul

exit /b %EXIT_CODE%
