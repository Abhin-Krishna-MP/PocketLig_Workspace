import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    allowedHosts: true,
    proxy: {
      '/targets': 'http://127.0.0.1:8008',
      '/jobs': 'http://127.0.0.1:8008',
    },
  },
  build: { chunkSizeWarningLimit: 6000 },
});
