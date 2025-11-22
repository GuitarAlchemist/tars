@echo off 
REM TARS Qwen3 Reasoning Monitor 
echo 📊 TARS Qwen3 Reasoning Monitor 
echo =============================== 
echo. 
echo 🔍 Available Qwen3 Models: 
ollama list | findstr qwen3 
echo. 
echo 💾 Model Storage Usage: 
dir "C:\Users\spare\.ollama\models" /s | find "qwen3" 
echo. 
echo 🧠 Reasoning Capabilities Status: 
type .tars\qwen3-config.json 
