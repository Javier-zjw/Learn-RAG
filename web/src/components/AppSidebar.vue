<script setup lang="ts">
import { computed, onBeforeUnmount, watch } from 'vue'
import { useRoute } from 'vue-router'
import { Plus, Monitor, Loading } from '@element-plus/icons-vue'
import { store, refreshKbs } from '../store'

const route = useRoute()
const activeId = computed(() => (route.path.startsWith('/kb/') ? String(route.params.id) : ''))

// 有知识库在建库时每 3 秒刷新一次，侧栏上的块数和进度标记跟着变
let timer: number | undefined
watch(
  () => store.kbs.some((kb) => kb.busy),
  (busy) => {
    window.clearInterval(timer)
    if (busy) timer = window.setInterval(refreshKbs, 3000)
  },
  { immediate: true },
)
onBeforeUnmount(() => window.clearInterval(timer))
</script>

<template>
  <aside class="sidebar">
    <RouterLink to="/" class="brand">
      <span class="logo"><i /><i /><i /></span>
      <span>Learn-RAG</span>
    </RouterLink>

    <RouterLink to="/new" custom v-slot="{ navigate }">
      <el-button type="primary" class="new-btn" :icon="Plus" @click="navigate">新建知识库</el-button>
    </RouterLink>

    <div class="group-label">知识库</div>
    <nav class="kb-list">
      <TransitionGroup name="list">
        <RouterLink v-for="kb in store.kbs" :key="kb.id" :to="`/kb/${kb.id}`" class="kb-item"
                    :class="{ active: kb.id === activeId }">
          <span class="kb-name">{{ kb.name }}</span>
          <span class="kb-meta">
            <el-icon v-if="kb.busy" class="spin"><Loading /></el-icon>
            {{ kb.files }} 文档 · {{ kb.chunks }} 块
          </span>
        </RouterLink>
      </TransitionGroup>
      <p v-if="store.kbsLoaded && !store.kbs.length" class="empty">还没有知识库</p>
    </nav>

    <RouterLink to="/env" class="env-link" :class="{ active: route.path === '/env' }">
      <el-icon><Monitor /></el-icon>
      <span>模型与环境</span>
      <span class="dot" :class="store.online ? 'on' : 'off'" :title="store.online ? '后端已连接' : '后端未连接'" />
    </RouterLink>
  </aside>
</template>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 18px 14px 14px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  min-height: 0;
}
.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 2px 6px 14px;
  font-weight: 650;
  font-size: 16px;
  color: var(--text);
}
.logo {
  display: inline-flex;
  flex-direction: column;
  justify-content: center;
  gap: 2.5px;
  width: 26px;
  height: 26px;
  padding: 0 6px;
  border-radius: 7px;
  background: var(--primary);
}
.logo i { display: block; height: 3px; border-radius: 2px; background: #fff; }
.logo i:nth-child(2) { width: 70%; opacity: 0.75; }
.logo i:nth-child(3) { width: 85%; opacity: 0.5; }
.new-btn { width: 100%; margin-bottom: 10px; }
.group-label {
  padding: 8px 8px 4px;
  font-size: 12px;
  color: var(--text-3);
}
.kb-list {
  position: relative;
  flex: 1;
  overflow-y: auto;
  min-height: 0;
}
.kb-item {
  position: relative;
  display: block;
  padding: 8px 10px 8px 12px;
  border-radius: 8px;
  color: var(--text);
  transition: background 0.18s ease;
}
.kb-item::before {
  content: '';
  position: absolute;
  left: 2px;
  top: 10px;
  bottom: 10px;
  width: 3px;
  border-radius: 2px;
  background: var(--primary);
  transform: scaleY(0);
  transition: transform 0.22s var(--ease);
}
.kb-item:hover { background: var(--surface-2); }
.kb-item.active { background: color-mix(in srgb, var(--primary) 9%, transparent); }
.kb-item.active::before { transform: scaleY(1); }
.kb-name {
  display: block;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  font-weight: 500;
}
.kb-meta {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--text-3);
}
.spin { animation: spin 1s linear infinite; color: var(--primary); }
.empty { padding: 4px 10px; color: var(--text-3); font-size: 13px; }
.env-link {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 10px;
  border-top: 1px solid var(--border);
  margin-top: 6px;
  color: var(--text-2);
  border-radius: 0 0 8px 8px;
  transition: color 0.18s ease;
}
.env-link:hover, .env-link.active { color: var(--primary); }
.dot { width: 7px; height: 7px; border-radius: 50%; margin-left: auto; }
.dot.on { background: var(--success); box-shadow: 0 0 0 3px color-mix(in srgb, var(--success) 18%, transparent); }
.dot.off { background: var(--danger); }
@media (max-width: 900px) {
  .sidebar { border-right: none; border-bottom: 1px solid var(--border); }
  .kb-list { max-height: 160px; }
}
</style>
