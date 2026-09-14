import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backendTarget = process.env.VITE_BACKEND_URL || 'http://localhost:8080'

export default defineConfig({
  plugins: [react()],
  server: {
    port: parseInt(process.env.VITE_PORT || '5174'),
    strictPort: false,
    host: true,
    open: false,
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
        secure: false,
      },
      '/demo-assets': {
        target: backendTarget,
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: backendTarget,
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
