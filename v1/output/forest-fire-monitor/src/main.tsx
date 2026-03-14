import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './styles/index.css'

// Initialize the application
console.log('🚀 Forest Fire Monitor - Initializing...');
console.log('🤖 Powered by TARS Autonomous Intelligence');

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
