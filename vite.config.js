import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({ plugins: [react(), VitePWA({ registerType: 'autoUpdate', manifest: { name: 'AnemiaScan', short_name: 'AnemiaScan', description: 'Non-invasive anaemia screening prototype', theme_color: '#0f6b5c', background_color: '#f6f8f7', display: 'standalone', start_url: '/' } })] });
