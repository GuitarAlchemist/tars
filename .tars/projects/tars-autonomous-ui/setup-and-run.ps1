# TARS Autonomous UI Setup and Run Script
# Created autonomously by TARS - handles all prerequisites and launches the UI

Write-Host "🤖 TARS AUTONOMOUS UI SETUP & LAUNCH" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔧 TARS is autonomously setting up its own UI..." -ForegroundColor Yellow
Write-Host ""

# Check if Node.js is installed
Write-Host "📋 Checking prerequisites..." -ForegroundColor Green
try {
    $nodeVersion = node --version 2>$null
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# Check if npm is available
try {
    $npmVersion = npm --version 2>$null
    Write-Host "✅ npm found: $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ npm not found. Please install npm." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📦 TARS is installing its own dependencies..." -ForegroundColor Yellow

# Install base dependencies
Write-Host "Installing React and TypeScript dependencies..." -ForegroundColor Cyan
npm install --silent

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install base dependencies" -ForegroundColor Red
    exit 1
}

# Install TARS-specific dependencies
Write-Host "Installing TARS-specific UI libraries..." -ForegroundColor Cyan
npm install zustand lucide-react clsx --silent

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install TARS dependencies" -ForegroundColor Red
    exit 1
}

# Install Tailwind CSS
Write-Host "Installing Tailwind CSS for TARS styling..." -ForegroundColor Cyan
npm install -D tailwindcss postcss autoprefixer --silent

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install Tailwind CSS" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎨 TARS is configuring its own styling..." -ForegroundColor Yellow

# Initialize Tailwind (if not already done)
if (-not (Test-Path "tailwind.config.js")) {
    Write-Host "Initializing Tailwind CSS configuration..." -ForegroundColor Cyan
    npx tailwindcss init -p --silent
}

Write-Host ""
Write-Host "✅ All dependencies installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 TARS is launching its autonomous UI..." -ForegroundColor Yellow
Write-Host ""
Write-Host "📱 TARS UI will be available at: http://localhost:5173" -ForegroundColor Cyan
Write-Host "🔧 TARS Control Center: Monitor autonomous operations" -ForegroundColor Cyan
Write-Host "🤖 Agent Dashboard: View multi-agent system status" -ForegroundColor Cyan
Write-Host "⚡ CUDA Metrics: Real-time performance monitoring" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop TARS UI server" -ForegroundColor Gray
Write-Host ""

# Start the development server
npm run dev
