import { defineConfig } from '@playwright/test'

/**
 * 最简 UI 冒烟测试配置。前提：后端已在 8000 端口跑起来（含真实 LLM），
 * 与 scripts/e2e_check.py 同一个前提，不在这里重复拉起后端 —— Agent 链路
 * 依赖真实 Ollama/Qwen，交给 Playwright webServer 管理会引入模型冷启动的
 * 不确定性，不如让运行者显式先启动好后端。
 */
export default defineConfig({
  testDir: './e2e',
  // 单条测试超时。写操作确认要走两轮完整链路（提问 + 确认执行），本机实测
  // /chat/confirm 单次就 186s（本地 7B 每次调用 7-15s，最坏 max_steps=6 轮 ×
  // 2 次调用）。原来的 150s 会在链路完全正常时超时失败，不是模型慢就是配置错，
  // 排查时容易误判成产品 bug。
  timeout: 400_000,
  // 两条流程都注册临时用户并写入同一个 SQLite 库，必须串行，避免把数据库锁/约束竞争
  // 误报为 UI 或 SSE 故障。
  workers: 1,
  fullyParallel: false,
  retries: 0,
  reporter: [
    ['list'],
    ['html', { outputFolder: process.env.COPILOT_E2E_REPORT_DIR ?? 'playwright-report', open: 'never' }],
  ],
  outputDir: process.env.COPILOT_E2E_OUTPUT_DIR ?? 'test-results',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:5173',
    reuseExistingServer: true,
    timeout: 30_000,
    env: {
      VITE_API_PROXY_TARGET: process.env.COPILOT_E2E_API ?? 'http://localhost:8000',
    },
  },
})
