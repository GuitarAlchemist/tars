@echo off 
REM TARS Qwen3 Reasoning Test Script 
echo 🧠 TARS Qwen3 Reasoning Test 
echo ========================== 
echo. 
echo 🔬 Testing Mathematical Reasoning... 
ollama run qwen3:30b-a3b "Solve step by step: Find the integral of 2x^3 - 4x + 1" 
echo. 
echo 🔬 Testing Logical Reasoning... 
ollama run qwen3:32b "Use logical reasoning: If all A are B, and all B are C, what can we conclude about A and C?" 
echo. 
echo 🔬 Testing Causal Reasoning... 
ollama run qwen3:14b "Analyze the causal chain: What factors might cause a software system to become slow over time?" 
echo. 
echo 🔬 Testing Strategic Reasoning... 
ollama run qwen3:8b "Develop a strategy: How should a software team prioritize bug fixes vs new features?" 
echo. 
echo ✅ Reasoning tests completed! 
