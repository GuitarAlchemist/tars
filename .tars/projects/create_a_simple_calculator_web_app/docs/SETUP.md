# Setup Guide

## Prerequisites
- Node.js (v14 or higher)
- Modern web browser
- Text editor (optional, for customization)

## Installation Steps

### 1. Quick Setup (Recommended)
```bash
# Simply run the provided script:
run.cmd
```

### 2. Manual Setup
```bash
# Navigate to project directory
cd create_a_simple_calculator_web_app

# Install dependencies
npm install

# Start development server
npm start
```

## Troubleshooting

### Node.js Not Found
- Download and install from: https://nodejs.org/
- Restart command prompt after installation

### Port Already in Use
- The app uses port 8080 by default
- If busy, the server will automatically find another port

### Browser Doesn't Open
- Manually navigate to: http://localhost:8080
- Check firewall settings if needed

## Development Mode
For development with auto-reload:
```bash
npx live-server --port=8080
```
