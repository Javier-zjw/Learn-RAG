import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发时前端跑在 5173，接口转发给 learn-rag serve（默认 8000）
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
  build: {
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes('element-plus')) return 'element-plus'
          if (id.includes('markdown-it') || id.includes('dompurify')) return 'markdown'
        },
      },
    },
  },
})
