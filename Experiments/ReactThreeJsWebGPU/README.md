# React + Three.js WebGPU Demo

This is a simple React application that uses Three.js with WebGPU rendering to display "Hello world" with an anodized blue texture and a light source with the same color as the sun.

## Features

- React for UI components
- Three.js for 3D rendering
- WebGPU rendering (with fallback to WebGL)
- Anodized blue texture for the "Hello World" text
- Sunlight-colored directional light
- Orbit controls for camera manipulation

## Prerequisites

- Node.js 16+ and npm
- A browser that supports WebGPU (Chrome 113+, Edge 113+, or other compatible browsers)

## Setup

1. Install dependencies:

```bash
npm install
```

2. Download the Inter-Bold.woff font and place it in the `public/fonts` directory.

3. Start the development server:

```bash
npm run dev
```

4. Build for production:

```bash
npm run build
```

## WebGPU Support

This application checks for WebGPU support in the browser. If WebGPU is not supported, it will display a message asking the user to use a compatible browser.

## Project Structure

- `src/App.tsx` - Main React component
- `src/components/HelloWorldScene.tsx` - Three.js scene component
- `src/styles/App.css` - CSS styles
- `public/fonts/` - Font files directory

## Notes

- The application uses the @react-three/fiber library to integrate Three.js with React
- The @react-three/drei library provides useful helpers for React Three Fiber
- The application uses Vite for fast development and building
