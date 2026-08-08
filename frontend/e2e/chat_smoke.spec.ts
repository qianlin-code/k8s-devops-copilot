import { expect, test } from '@playwright/test'

/**
 * 最简 UI 冒烟：打开首页 -> 提问 -> 等待流式对话结束（进度指示消失）。
 * 不断言回答内容——本地 7B 模型的生成结果不稳定，这里只验证 SSE 全链路
 * （前端渲染 -> 后端 chat/stream -> Agent -> done 事件）没有断裂。
 *
 * 前提：后端已在 8000 端口运行（真实 Ollama），知识库已灌入至少一篇文档
 * （scripts/seed_kb.py），且浏览器 localStorage 里的 API Key 与后端 .env
 * 一致。CI 环境下这三项都需要在跑测试前准备好。
 */
test('提问后能收到回答，进度指示器最终消失', async ({ page }) => {
  await page.goto('/')

  // /health 端点不需要鉴权（见 README「生产环境配置护栏」），所以健康状态
  // 显示在线并不代表 API Key 已配置——必须无条件走一次设置流程写入 API Key，
  // 否则后续 /chat/stream 会立刻 401，整轮"对话"在 1s 内假通过。
  await page.getByRole('button', { name: '设置' }).click()
  await page.getByPlaceholder('与后端 .env 的 API_KEY 一致').fill('dev-local-api-key-change-me')
  await page.getByRole('button', { name: '保存并刷新' }).click()
  await expect(page.locator('.status.err')).not.toBeVisible({ timeout: 10_000 })

  const composer = page.locator('textarea')
  await composer.fill('登录提示 403 是什么原因')
  await page.getByRole('button', { name: '发送' }).click()

  // 用户消息立即出现在气泡列表里
  await expect(page.locator('.msg-user').last()).toContainText('登录提示 403 是什么原因')

  // 流式对话结束的标志：「取消」按钮消失，「发送」按钮重新出现。
  // 本地 7B 模型一轮对话串联 4 次 LLM 调用，冷启动后仍要数秒到数十秒，
  // 若这一步在 2s 内就完成，大概率是请求提前失败（鉴权/网络错误），不是真的跑完
  const start = Date.now()
  await expect(page.getByRole('button', { name: '发送' })).toBeVisible({
    timeout: 120_000,
  })
  expect(Date.now() - start).toBeGreaterThan(2_000)

  // 必须是成功的回复，不能是失败态（失败消息复用同一个 .msg-assistant class，
  // 只是多一个 .failed 修饰符，不能靠"存在 assistant 消息"来判断成功）
  const lastAssistant = page.locator('.msg-assistant').last()
  await expect(lastAssistant).toBeVisible()
  await expect(lastAssistant.locator('.msg-text.failed')).toHaveCount(0)
})
