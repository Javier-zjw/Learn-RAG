<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, RefreshRight, Delete, Search, View, EditPen } from '@element-plus/icons-vue'
import { api, watchJob, type DocRow, type Job, type KbInfo, type Settings } from '../api'
import { STATUS, bytes, chunkerSummary, clone, encoderSummary, settingsError, timeAgo, typeGroup } from '../format'
import { loadEnv, refreshKbs, store } from '../store'
import JobPanel from '../components/JobPanel.vue'
import FileDrop, { type Entry } from '../components/FileDrop.vue'
import SettingsForm from '../components/SettingsForm.vue'

const route = useRoute()
const router = useRouter()
const kbId = computed(() => String(route.params.id))
const info = ref<KbInfo | null>(null)
const job = ref<Job | null>(null)
const showJob = ref(true)
const error = ref('')
const query = ref('')
let unwatch: (() => void) | null = null

async function load() {
  try {
    info.value = await api.kb(kbId.value)
    error.value = ''
    // 打开页面时有任务在跑就跟上它；刚从新建页跳过来（?job=）时即使已经跑完也展示结果；其他历史任务不展示
    const latest = info.value.job
    const running = latest && latest.status !== 'done' && latest.status !== 'failed'
    if (latest && (running || latest.id === route.query.job) && latest.id !== job.value?.id) follow(latest)
  } catch (err) {
    error.value = (err as Error).message
  }
}

function follow(next: Job) {
  unwatch?.()
  job.value = next
  showJob.value = true
  unwatch = watchJob(next.id, (update) => {
    const finished = update.status === 'done' || update.status === 'failed'
    // 某个文件刚处理完时刷新文档表（块数会变）；任务结束后刷新整页和侧栏
    const before = job.value?.processed ?? 0
    job.value = update
    if (finished) {
      load()
      refreshKbs()
    } else if (update.processed !== before) {
      load()
    }
  })
}

watch(kbId, () => {
  unwatch?.()
  job.value = null
  load()
}, { immediate: true })
onBeforeUnmount(() => unwatch?.())

const busy = computed(() => !!job.value && (job.value.status === 'queued' || job.value.status === 'running'))

// 文档表：正在处理的文件显示任务里的实时状态
const rows = computed<DocRow[]>(() => {
  const live = new Map((busy.value ? job.value?.files ?? [] : []).map((f) => [f.path, f]))
  const q = query.value.trim().toLowerCase()
  return (info.value?.documents ?? [])
    .filter((d) => !q || d.doc_id.toLowerCase().includes(q))
    .map((d) => {
      const f = live.get(d.doc_id)
      return f ? { ...d, status: f.status, message: f.message } : d
    })
})

const verifyTag = computed(() => {
  const v = info.value?.verify
  if (!v) return { text: '未检查', type: 'info' as const }
  return v.ok ? { text: '通过', type: 'success' as const } : { text: `${v.problems.length} 个问题`, type: 'danger' as const }
})

function open(row: DocRow) {
  if (row.chunks > 0) router.push(`/kb/${kbId.value}/doc/${row.doc_id}`)
}

async function removeDoc(row: DocRow) {
  try {
    await api.deleteDocument(kbId.value, row.doc_id)
    ElMessage.success(`已删除 ${row.doc_id}`)
    await load()
    refreshKbs()
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}

async function removeKb() {
  try {
    await ElMessageBox.confirm(`删除知识库「${info.value?.name}」？上传的文件、向量和分块都会被删除，不能恢复。`, '删除知识库',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消', confirmButtonClass: 'el-button--danger' })
  } catch {
    return
  }
  try {
    await api.deleteKb(kbId.value)
    ElMessage.success('已删除')
    await refreshKbs()
    router.push('/')
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}

// ---------------------------------------------------------------- 改名（名称只用于显示，随时可改）
const editing = ref(false)
const draftName = ref('')
const nameInput = ref<{ focus: () => void; select: () => void }>()
async function startRename() {
  draftName.value = info.value?.name ?? ''
  editing.value = true
  await nextTick()
  nameInput.value?.focus()
  nameInput.value?.select()
}
async function saveRename() {
  if (!editing.value) return
  editing.value = false
  const name = draftName.value.trim()
  if (!info.value || !name || name === info.value.name) return
  try {
    const summary = await api.renameKb(kbId.value, name)
    info.value.name = summary.name
    refreshKbs()
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}

// ---------------------------------------------------------------- 追加文件
const appendOpen = ref(false)
const appendId = ref('')
const appendEntries = ref<Entry[]>([])
const appendReady = computed(() => appendEntries.value.some((e) => e.state === 'ready')
  && !appendEntries.value.some((e) => e.state === 'uploading'))
function openAppend() {
  appendId.value = ''
  appendEntries.value = []
  appendOpen.value = true
}
async function submitAppend() {
  try {
    follow(await api.addFiles(kbId.value, appendId.value))
    appendOpen.value = false
    load()
    refreshKbs()
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}

// ---------------------------------------------------------------- 重建
const rebuildOpen = ref(false)
const rebuildSettings = ref<Settings | null>(null)
const resets = computed(() => {
  const now = rebuildSettings.value
  const old = info.value?.settings
  if (!now || !old) return false
  return JSON.stringify([now.encoder, now.index]) !== JSON.stringify([old.encoder, old.index])
})
async function openRebuild() {
  await loadEnv()
  rebuildSettings.value = clone(info.value!.settings)
  rebuildOpen.value = true
}
async function submitRebuild() {
  try {
    follow(await api.rebuild(kbId.value, rebuildSettings.value!))
    rebuildOpen.value = false
    load()
    refreshKbs()
  } catch (err) {
    ElMessage.error((err as Error).message)
  }
}
</script>

<template>
  <div class="page">
    <el-result v-if="error && !info" icon="warning" title="打不开这个知识库" :sub-title="error">
      <template #extra><el-button @click="router.push('/')">返回</el-button></template>
    </el-result>

    <template v-else-if="info">
      <div class="page-head">
        <div class="head-main">
          <el-input v-if="editing" ref="nameInput" v-model="draftName" maxlength="40" size="large" class="name-edit"
                    @keyup.enter="saveRename" @keyup.esc="editing = false" @blur="saveRename" />
          <h1 v-else class="title" title="点击修改名称" @click="startRename">
            <span>{{ info.name }}</span>
            <el-icon class="edit-icon"><EditPen /></el-icon>
          </h1>
          <div class="tags">
            <el-tag effect="plain" round>{{ chunkerSummary(info.settings) }}</el-tag>
            <el-tag effect="plain" round>{{ encoderSummary(info.settings) }}</el-tag>
            <el-tag effect="plain" round>{{ info.settings.index.type === 'chroma' ? 'Chroma' : 'Flat' }}</el-tag>
          </div>
        </div>
        <div class="actions">
          <el-button :icon="Plus" @click="openAppend">追加文件</el-button>
          <el-button :icon="RefreshRight" :disabled="busy" @click="openRebuild">重建</el-button>
          <el-button :icon="Delete" :disabled="busy" @click="removeKb">删除</el-button>
        </div>
      </div>

      <div class="stats">
        <div class="stat rise" style="--i: 0"><span>文档</span><b>{{ info.files }}</b></div>
        <div class="stat rise" style="--i: 1"><span>子块</span><b>{{ info.chunks }}</b></div>
        <div class="stat rise" style="--i: 2"><span>父块</span><b>{{ info.parents }}</b></div>
        <div class="stat rise" style="--i: 3">
          <span>完整性</span>
          <b><el-tag :type="verifyTag.type" effect="light" round>{{ verifyTag.text }}</el-tag></b>
        </div>
      </div>

      <Transition name="slide">
        <JobPanel v-if="job && showJob" :job="job" @close="showJob = false" />
      </Transition>

      <section class="card rise" style="--i: 4">
        <div class="card-head">
          <h3>文档</h3>
          <el-input v-model="query" :prefix-icon="Search" placeholder="搜索文件名" clearable class="search" />
        </div>
        <el-table :data="rows" row-key="doc_id" class="docs" @row-click="open" empty-text="还没有文档">
          <el-table-column label="文件" min-width="260">
            <template #default="{ row }">
              <div class="file-cell" :class="{ link: row.chunks > 0 }">
                <span class="type-badge" :class="`type-${typeGroup(row.type)}`">{{ row.type }}</span>
                <span class="file-name" :title="row.doc_id">{{ row.doc_id }}</span>
              </div>
            </template>
          </el-table-column>
          <el-table-column label="状态" width="110">
            <template #default="{ row }">
              <el-tooltip :disabled="!row.message" :content="row.message" placement="top">
                <el-tag size="small" :type="STATUS[row.status as keyof typeof STATUS]?.type ?? 'info'" effect="light">
                  {{ STATUS[row.status as keyof typeof STATUS]?.label ?? row.status }}
                </el-tag>
              </el-tooltip>
            </template>
          </el-table-column>
          <el-table-column label="解析器" width="100">
            <template #default="{ row }"><span class="muted">{{ row.parser || '—' }}</span></template>
          </el-table-column>
          <el-table-column prop="chunks" label="子块" width="76" align="right" />
          <el-table-column prop="parents" label="父块" width="76" align="right" />
          <el-table-column label="页数" width="70" align="right">
            <template #default="{ row }">{{ row.pages ?? '—' }}</template>
          </el-table-column>
          <el-table-column label="大小" width="90" align="right">
            <template #default="{ row }">{{ bytes(row.size) }}</template>
          </el-table-column>
          <el-table-column label="更新" width="110">
            <template #default="{ row }"><span class="muted">{{ timeAgo(row.updated_at ?? row.uploaded_at) }}</span></template>
          </el-table-column>
          <el-table-column label="" width="96" align="right">
            <template #default="{ row }">
              <div class="row-actions" @click.stop>
                <el-button text circle size="small" :icon="View" :disabled="!row.chunks" @click="open(row)" />
                <el-popconfirm :title="`删除 ${row.doc_id}？`" confirm-button-text="删除" cancel-button-text="取消"
                               width="240" @confirm="removeDoc(row)">
                  <template #reference>
                    <el-button text circle size="small" :icon="Delete" :disabled="busy" />
                  </template>
                </el-popconfirm>
              </div>
            </template>
          </el-table-column>
        </el-table>
      </section>
    </template>
    <el-skeleton v-else :rows="10" animated />

    <el-dialog v-model="appendOpen" title="追加文件" width="640px" destroy-on-close>
      <p class="hint dialog-hint">沿用这个知识库的参数。同名文件会替换旧版本；内容没变的文件自动跳过，不会重复向量化。</p>
      <FileDrop v-model:upload-id="appendId" v-model:entries="appendEntries" compact />
      <template #footer>
        <el-button @click="appendOpen = false">取消</el-button>
        <el-button type="primary" :disabled="!appendReady" @click="submitAppend">开始处理</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="rebuildOpen" title="按新参数重建" width="520px" destroy-on-close>
      <SettingsForm v-if="rebuildSettings && store.env" v-model="rebuildSettings" :env="store.env" />
      <el-alert :type="resets ? 'warning' : 'info'" :closable="false" show-icon class="rebuild-note"
                :title="resets ? '换了 embedding 模型或向量库参数：会清空片段库，全部文件重新向量化'
                  : '只改切分参数时逐篇替换旧分块；参数没变时只处理上次失败或中断的文件'" />
      <template #footer>
        <el-button @click="rebuildOpen = false">取消</el-button>
        <el-button type="primary" :disabled="!rebuildSettings || !!settingsError(rebuildSettings)"
                   @click="submitRebuild">开始重建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.head-main { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.title { display: flex; align-items: center; gap: 8px; min-width: 0; cursor: pointer; }
.title span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.edit-icon { flex: none; font-size: 16px; color: var(--text-3); opacity: 0; transition: opacity 0.15s ease, color 0.15s ease; }
.title:hover .edit-icon { opacity: 1; }
.title:hover .edit-icon:hover { color: var(--primary); }
.name-edit { max-width: 420px; }
.name-edit :deep(.el-input__inner) { font-size: 18px; font-weight: 600; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; }
.actions { display: flex; gap: 4px; flex: none; }
.stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 18px;
}
.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 12px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.stat span { font-size: 12px; color: var(--text-3); }
.stat b { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.3; }
.search { width: 220px; }
.docs :deep(.el-table__row) { cursor: default; transition: background 0.15s ease; }
.file-cell { display: flex; align-items: center; gap: 10px; min-width: 0; }
.file-cell.link { cursor: pointer; }
.file-cell.link .file-name { transition: color 0.15s ease; }
.docs :deep(.el-table__row:hover) .file-cell.link .file-name { color: var(--primary); }
.file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row-actions { display: inline-flex; }
.dialog-hint { margin: -6px 0 14px; }
.rebuild-note { margin-top: 12px; }
@media (max-width: 760px) {
  .stats { grid-template-columns: repeat(2, 1fr); }
  .page-head { flex-direction: column; align-items: flex-start; }
}
</style>
