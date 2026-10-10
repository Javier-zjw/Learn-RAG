<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { CircleCheckFilled, CircleCloseFilled, Refresh } from '@element-plus/icons-vue'
import { loadEnv, store } from '../store'

const loading = ref(false)
async function reload() {
  loading.value = true
  await loadEnv(true)
  loading.value = false
}
onMounted(reload)
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1>模型与环境</h1>
        <div class="sub">只检查装没装、配没配，不显示密钥的值。修改 .env 或安装依赖后点"刷新"，无需重启页面</div>
      </div>
      <el-button :icon="Refresh" :loading="loading" @click="reload">刷新</el-button>
    </div>

    <el-result v-if="!store.env && !loading" icon="warning" title="后端未连接"
               sub-title="请在项目根目录运行 learn-rag serve" />

    <div v-else-if="store.env" class="grid">
      <section class="card rise" style="--i: 0">
        <div class="card-head"><h3>Embedding 模型</h3></div>
        <div class="card-body list">
          <div v-for="e in store.env.encoders" :key="e.type" class="item" :class="{ off: !e.available }">
            <el-icon :class="e.available ? 'good' : 'bad'">
              <CircleCheckFilled v-if="e.available" /><CircleCloseFilled v-else />
            </el-icon>
            <div>
              <div class="name">{{ e.label }} <span v-if="e.available && e.model" class="muted mono">{{ e.model }}</span></div>
              <div class="hint">{{ e.note }}</div>
            </div>
          </div>
        </div>
      </section>

      <section class="card rise" style="--i: 1">
        <div class="card-head"><h3>环境变量（.env）</h3></div>
        <div class="card-body list">
          <div v-for="k in store.env.keys" :key="k.name" class="item">
            <el-icon :class="k.configured ? 'good' : 'idle'">
              <CircleCheckFilled v-if="k.configured" /><CircleCloseFilled v-else />
            </el-icon>
            <div>
              <div class="name mono">{{ k.name }}</div>
              <div class="hint">{{ k.label }} · {{ k.configured ? '已配置' : '未配置' }}</div>
            </div>
          </div>
        </div>
      </section>

      <section class="card rise wide" style="--i: 2">
        <div class="card-head"><h3>解析与存储依赖</h3></div>
        <div class="card-body packages">
          <div v-for="p in store.env.packages" :key="p.name" class="item" :class="{ off: !p.installed }">
            <el-icon :class="p.installed ? 'good' : 'idle'">
              <CircleCheckFilled v-if="p.installed" /><CircleCloseFilled v-else />
            </el-icon>
            <div>
              <div class="name">{{ p.name }}</div>
              <div class="hint">
                <template v-if="p.installed">已安装</template>
                <template v-else>未安装 · <code>{{ p.install }}</code></template>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.wide { grid-column: 1 / -1; }
.list { display: flex; flex-direction: column; gap: 14px; }
.packages { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px 24px; }
.item { display: flex; gap: 10px; align-items: flex-start; }
.item .el-icon { margin-top: 3px; font-size: 16px; flex: none; }
.item.off .name { color: var(--text-2); }
.name { font-weight: 500; }
.good { color: var(--success); }
.bad { color: var(--danger); }
.idle { color: var(--text-3); }
@media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
</style>
