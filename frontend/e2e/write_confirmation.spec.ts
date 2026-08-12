import { expect, test } from '@playwright/test'

/**
 * 验证写操作确认流程：提问触发写工具 → 弹确认卡片 → 点确认 → 执行并返回结果。
 *
 * 前提：后端已在 8000 端口运行，知识库已灌入 K8s 文档，且 mock 数据已初始化
 * （backend/data/app.db 里有 ops-demo/worker-queue，它是唯一 available_replicas>0
 * 的 Deployment，其余会先撞上「副本全不可用不许重启」的业务规则）。
 *
 * 耗时说明：确认执行会再走一遍「检索 + 执行写操作 + 路由/充分性校验循环」，
 * 本机实测 `/chat/confirm` 单次 **186 秒**（本地 7B 每次调用 7-15s，最坏
 * max_steps=6 轮 × 2 次调用）。所以这里的等待给到 240s、配置里的单测超时给到
 * 400s —— 早前设 120s 会在链路完全正常的情况下超时失败。
 *
 * 断言只看结构（确认卡片消失、无失败态、留下写操作记录），不匹配模型措辞：
 * 本地 7B 的具体用词不稳定，断言原文会变成常态性 flaky。失败路径（确认后
 * 工具报错）由 `tests/contract/test_contract_chat.py::
 * test_confirmed_write_failure_reports_error_not_another_confirmation`
 * 用替身模型确定性地覆盖，不在这里重复跑真实模型。
 */
test('写操作确认流程：触发→确认→执行', async ({ page }, testInfo) => {
  await page.goto('/')

  const username = `e2e-write-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  const password = `e2e-${crypto.randomUUID()}`
  const organization = `E2E Write Org ${crypto.randomUUID()}`
  await page.getByRole('tab', { name: '注册' }).click()
  await page.getByLabel('用户名').fill(username)
  await page.getByLabel('密码').fill(password)
  await page.getByLabel('组织名称').fill(organization)
  await page.getByRole('button', { name: '注册并登录' }).click()
  await expect(page.locator('textarea')).toBeVisible({ timeout: 10_000 })

  const composer = page.locator('textarea')
  await composer.fill('ops-demo 下 worker-queue 这个 Deployment 配置已经修好了，帮我重启一下')
  await page.getByRole('button', { name: '发送' }).click()

  // 等待流式对话结束并出现确认卡片
  await expect(page.getByRole('button', { name: '发送' })).toBeVisible({ timeout: 240_000 })

  // 确认卡片应该出现（包含工具名和"确认执行"按钮）
  const confirmCard = page.locator('.confirm-card')
  await expect(confirmCard).toBeVisible()
  await expect(confirmCard).toContainText('restart_deployment')

  // 点击"确认执行"
  await confirmCard.getByRole('button', { name: '确认执行' }).click()

  // 再次等待流式响应结束。240s 不是保守取值，是按实测 186s 留的余量
  await expect(page.getByRole('button', { name: '发送' })).toBeVisible({ timeout: 240_000 })

  // 确认卡片必须消失：留着就意味着又要求确认一次，用户会陷入死循环
  await expect(page.locator('.confirm-card')).toHaveCount(0)

  // 最后一条 assistant 消息不能是失败态
  const lastAssistant = page.locator('.msg-assistant').last()
  await expect(lastAssistant).toBeVisible()
  await expect(lastAssistant.locator('.msg-text.failed')).toHaveCount(0)

  // 写操作确实执行过：执行链路里有 execute_confirmed_write 节点。
  // 比断言回答措辞可靠——trace 是结构化数据，不受模型用词影响。
  await lastAssistant.getByRole('button', { name: '执行链路' }).click()
  await expect(lastAssistant.locator('.step-execute_confirmed_write')).toHaveCount(1)
  await page.screenshot({ path: testInfo.outputPath('write-confirmation-success.png'), fullPage: true })
})
