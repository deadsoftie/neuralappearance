@echo off
setlocal

python "%~dp0configure.py" install %*

exit /b 0
