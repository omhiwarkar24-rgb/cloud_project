@echo off
title Live Node Monitoring Agent
echo ================================================================
echo           LIVE NODE MONITORING AGENT LAUNCHER
echo ================================================================
echo.

set /p SERVER_IP="Enter Central Server Laptop IP (default: 10.175.163.77): "
if "%SERVER_IP%"=="" set SERVER_IP=10.175.163.77

echo.
echo [STARTING] Connecting to Central Server at http://%SERVER_IP%:8000 ...
echo Press Ctrl+C in this window at any time to stop the agent.
echo.
python collector.py --server-url http://%SERVER_IP%:8000
pause
