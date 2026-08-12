import { expect, test } from '@playwright/test'

/**
 * 最简 UI 冒烟：打开首页 -> 提问 -> 等待流式对话结束（进度指示消失）。
 * 不断言回答内容——本地 7B 模型的生成结果不稳定，这里只验证 SSE 全链路
 * （前端渲染 -> 后端 chat/stream -> Agent -> done 事件）没有断裂。
 *
 * 前提：后端已在 8000 端口运行（真实 Ollama），知识库已灌入至少一篇文档
 * （scripts/seed_kb.py）。测试自行注册临时账号并通过 JWT 登录。
 *
 * 数据说明：本测试通过真实前端发起对话，会在后端的 data/app.db 里留下一条
 * 真实会话记录，与 scripts/e2e_check.py 同理——项目没有 DELETE /conversations
 * 端点（越权删除他人会话的风险不值得为测试便利新开一个），所以无法自动清理。
 * 若不想污染 dev 库，让后端指向独立的 DATABASE_URL（改 vite.config.ts 里的
 * proxy target 指向那个实例），或接受会话记录留存——它们和真实用户数据
 * 结构一致，不影响其他功能，只是会让「历史记录」页面多几条测试对话。
 */
test('提问后能收到回答，进度指示器最终消失', async ({ page }, testInfo) => {
  await page.goto('/')

  const username = `e2e-chat-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  const password = `e2e-${crypto.randomUUID()}`
  const organization = `E2E Chat Org ${crypto.randomUUID()}`
  await page.getByRole('tab', { name: '注册' }).click()
  await page.getByLabel('用户名').fill(username)
  await page.getByLabel('密码').fill(password)
  await page.getByLabel('组织名称').fill(organization)
  await page.getByRole('button', { name: '注册并登录' }).click()
  await expect(page.locator('textarea')).toBeVisible({ timeout: 10_000 })

  // 用知识性问题而不是工具调用问题：工具路径要多走「执行工具 + 充分性校验 +
  // 再路由」若干轮，本机实测会超过 120s，放在冒烟测试里只会变成常态性超时失败。
  // 工具与写操作路径由 write_confirmation.spec.ts 端到端覆盖，这里只验
  // 「前端渲染 → SSE → Agent → done」这条链路没断。
  const composer = page.locator('textarea')
  await composer.fill('Pod 一直是 Pending 状态该怎么排查')
  await page.getByRole('button', { name: '发送' }).click()

  // 用户消息立即出现在气泡列表里
  await expect(page.locator('.msg-user').last()).toContainText('Pending')

  // 流式对话结束的标志：「取消」按钮消失，「发送」按钮重新出现。
  // 本地 7B 模型一轮对话串联 4 次 LLM 调用，实测知识性问题 70s 量级，
  // 给到 180s 留余量；若这一步在 2s 内就完成，大概率是请求提前失败，
  // 不是真的跑完。
  const start = Date.now()
  await expect(page.getByRole('button', { name: '发送' })).toBeVisible({
    timeout: 180_000,
  })
  expect(Date.now() - start).toBeGreaterThan(2_000)

  // 必须是成功的回复，不能是失败态（失败消息复用同一个 .msg-assistant class，
  // 只是多一个 .failed 修饰符，不能靠"存在 assistant 消息"来判断成功）
  const lastAssistant = page.locator('.msg-assistant').last()
  await expect(lastAssistant).toBeVisible()
  await expect(lastAssistant.locator('.msg-text.failed')).toHaveCount(0)
  await page.screenshot({ path: testInfo.outputPath('chat-success.png'), fullPage: true })
})
