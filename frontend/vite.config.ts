import { fileURLToPath, URL } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        // 本地真实验收用独立 Compose project 时会覆写到隔离端口，
        // 日常开发仍保持默认 8000，避免测试数据写入开发实例。
        target: apiProxyTarget,
        changeOrigin: true,
        // Agent 链路本地实测 20-40s，默认代理超时会在中途掐断连接，
        // 前端看到 ERR_ABORTED 而后端仍在处理，用户会误判失败并重复提交。
        timeout: 300_000,
        proxyTimeout: 300_000,
        // SSE 必须关掉缓冲，否则事件会被攒到最后一起吐出，失去流式意义
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              delete proxyRes.headers['content-length']
            }
          })
        },
      },
    },
  },
})
