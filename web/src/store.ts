// 跨页面共享的少量状态：知识库列表（侧栏）和环境信息（表单默认值、可选模型）

import { reactive } from 'vue'
import { api, type EnvReport, type KbSummary } from './api'

export const store = reactive({
  kbs: [] as KbSummary[],
  kbsLoaded: false,
  env: null as EnvReport | null,
  online: true,
})

export async function refreshKbs(): Promise<void> {
  try {
    store.kbs = await api.kbs()
    store.online = true
  } catch {
    store.online = false
  }
  store.kbsLoaded = true
}

export async function loadEnv(force = false): Promise<EnvReport | null> {
  if (store.env && !force) return store.env
  try {
    store.env = await api.env()
    store.online = true
  } catch {
    store.online = false
  }
  return store.env
}
