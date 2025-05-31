#!/bin/bash

# TARS Autonomous UI Setup and Run Script
# Created autonomously by TARS - handles all prerequisites and launches the UI

echo "🤖 TARS AUTONOMOUS UI SETUP & LAUNCH"
echo "===================================="
echo ""
echo "🔧 TARS is autonomously setting up its own UI..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Node.js is installed
echo -e "${GREEN}📋 Checking prerequisites...${NC}"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✅ Node.js found: $NODE_VERSION${NC}"
else
    echo -e "${RED}❌ Node.js not found. Please install Node.js from https://nodejs.org/${NC}"
    exit 1
fi

# Check if npm is available
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✅ npm found: $NPM_VERSION${NC}"
else
    echo -e "${RED}❌ npm not found. Please install npm.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}📦 TARS is installing its own dependencies...${NC}"

# Install base dependencies
echo -e "${CYAN}Installing React and TypeScript dependencies...${NC}"
npm install --silent

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install base dependencies${NC}"
    exit 1
fi

# Install TARS-specific dependencies
echo -e "${CYAN}Installing TARS-specific UI libraries...${NC}"
npm install zustand lucide-react clsx --silent

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install TARS dependencies${NC}"
    exit 1
fi

# Install Tailwind CSS
echo -e "${CYAN}Installing Tailwind CSS for TARS styling...${NC}"
npm install -D tailwindcss postcss autoprefixer --silent

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install Tailwind CSS${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}🎨 TARS is configuring its own styling...${NC}"

# Initialize Tailwind (if not already done)
if [ ! -f "tailwind.config.js" ]; then
    echo -e "${CYAN}Initializing Tailwind CSS configuration...${NC}"
    npx tailwindcss init -p > /dev/null 2>&1
fi

echo ""
echo -e "${GREEN}✅ All dependencies installed successfully!${NC}"
echo ""
echo -e "${YELLOW}🚀 TARS is launching its autonomous UI...${NC}"
echo ""
echo -e "${CYAN}📱 TARS UI will be available at: http://localhost:5173${NC}"
echo -e "${CYAN}🔧 TARS Control Center: Monitor autonomous operations${NC}"
echo -e "${CYAN}🤖 Agent Dashboard: View multi-agent system status${NC}"
echo -e "${CYAN}⚡ CUDA Metrics: Real-time performance monitoring${NC}"
echo ""
echo -e "${NC}Press Ctrl+C to stop TARS UI server${NC}"
echo ""

# Start the development server
npm run dev
