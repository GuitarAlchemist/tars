import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  define: {
    'process.env': {}
  },
  server: {
    port: 3000,
    open: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          arcgis: ['@arcgis/core', '@arcgis/map-components-react'],
          ui: ['@mui/material', '@mui/icons-material']
        }
      }
    }
  },
  optimizeDeps: {
    include: ['@arcgis/core', '@arcgis/map-components-react']
  },
  resolve: {
    alias: {
      '@': '/src'
    }
  }
})
