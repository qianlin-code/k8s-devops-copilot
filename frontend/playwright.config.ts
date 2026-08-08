import { defineConfig } from '@playwright/test'

/**
 * 最简 UI 冒烟测试配置。前提：后端已在 8000 端口跑起来（含真实 LLM），
 * 与 scripts/e2e_check.py 同一个前提，不在这里重复拉起后端 —— Agent 链路
 * 依赖真实 Ollama/Qwen，交给 Playwright webServer 管理会引入模型冷启动的
 * 不确定性，不如让运行者显式先启动好后端。
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 150_000, // 单条测试超时：留够本地 7B 模型冷启动 + 一轮对话的时间
  fullyParallel: false,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 30_000,
  },
})
