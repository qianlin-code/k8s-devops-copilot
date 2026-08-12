import { expect, test } from '@playwright/test'

const USER_STORAGE = 'copilot.currentUser'
const TOKEN_STORAGE = 'copilot.accessToken'

async function mockHealth(page: import('@playwright/test').Page) {
  await page.route('**/api/v1/health', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ status: 'ok', environment: 'test' }),
    })
  })
}

async function mockEmptyHistory(page: import('@playwright/test').Page) {
  await page.route('**/api/v1/conversations?limit=50', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ items: [], total: 0 }),
    })
  })
}

async function signInAs(
  page: import('@playwright/test').Page,
  role: 'user' | 'admin',
) {
  await page.addInitScript(({ userKey, tokenKey, currentRole }) => {
    localStorage.setItem(tokenKey, 'test-only-token')
    localStorage.setItem(userKey, JSON.stringify({
      user_id: 'test-user-id',
      username: `test-${currentRole}`,
      role: currentRole,
      organization_id: 'test-organization-id',
    }))
  }, { userKey: USER_STORAGE, tokenKey: TOKEN_STORAGE, currentRole: role })
}

test('未登录用户只能看到认证入口', async ({ page }) => {
  await mockHealth(page)

  await page.goto('/chat')

  await expect(page.getByRole('tab', { name: '登录' })).toBeVisible()
  await expect(page.locator('textarea')).toHaveCount(0)
})

test('普通用户访问知识库会重定向到聊天页，历史和会话 URL 可解析', async ({ page }) => {
  await mockHealth(page)
  await mockEmptyHistory(page)
  await page.route('**/api/v1/conversations/test-conversation-id?include_trace=true', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ conversation_id: 'test-conversation-id', messages: [] }),
    })
  })
  await signInAs(page, 'user')

  await page.goto('/knowledge')
  await expect(page).toHaveURL(/\/chat$/)
  await expect(page.getByRole('heading', { name: '对话' })).toBeVisible()
  await expect(page.getByRole('link', { name: '知识库' })).toHaveCount(0)

  await page.goto('/history')
  await expect(page).toHaveURL(/\/history$/)
  await expect(page.getByRole('heading', { name: '会话记录' })).toBeVisible()

  await page.goto('/chat/test-conversation-id')
  await expect(page).toHaveURL(/\/chat\/test-conversation-id$/)
  await expect(page.getByRole('heading', { name: '对话' })).toBeVisible()
})

test('管理员可进入知识库页面', async ({ page }) => {
  await mockHealth(page)
  await signInAs(page, 'admin')
  await page.route('**/api/v1/knowledge/documents', async (route) => {
    await route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        collection_name: 'test-collection', total: 0, vector_count: 0, bm25_index_size: 0, documents: [],
      }),
    })
  })
  await page.route('**/api/v1/knowledge/sedimentations?status=pending', async (route) => {
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify({ total: 0, entries: [] }) })
  })

  await page.goto('/knowledge')

  await expect(page).toHaveURL(/\/knowledge$/)
  await expect(page.getByRole('heading', { name: '知识库管理' })).toBeVisible()
})
