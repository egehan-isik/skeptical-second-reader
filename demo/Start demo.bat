@echo off
rem Opens the Skeptical Second Reader demo in the browser. Double-click to start;
rem close this window to stop it. Paths are relative to this file.
title Skeptical Second Reader demo
cd /d "%~dp0.."

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install it from https://docs.astral.sh/uv/ and try again.
    pause
    exit /b 1
)

rem The stored answers work without the local model; live calls need Ollama.
curl -s -o nul http://localhost:11434/api/tags
if errorlevel 1 (
    where ollama >nul 2>nul
    if not errorlevel 1 (
        echo Starting the local model server...
        start "Ollama" /min ollama serve
    ) else (
        echo Ollama was not found: the demo shows the stored answers only.
    )
)

echo Opening the demo in the browser...
uv run streamlit run app/streamlit_app.py
pause
