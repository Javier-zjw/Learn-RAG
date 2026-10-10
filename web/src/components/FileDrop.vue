<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { UploadFilled, Document, FolderOpened, Close, Loading, CircleCheckFilled, WarningFilled } from '@element-plus/icons-vue'
import { api } from '../api'
import { bytes, typeGroup } from '../format'

export interface Entry {
  path: string
  size: number
  type: string
  state: 'uploading' | 'ready' | 'unsupported' | 'error'
  message?: string
}

// 选中文件后立刻上传到暂存区：一键推荐要扫描真实文件，建库时直接从暂存区移入知识库
const uploadId = defineModel<string>('uploadId', { default: '' })
const entries = defineModel<Entry[]>('entries', { default: () => [] })
defineProps<{ compact?: boolean }>()

const BATCH = 8
const dragging = ref(false)
const fileInput = ref<HTMLInputElement>()
const folderInput = ref<HTMLInputElement>()
let queue: Promise<void> = Promise.resolve()

onMounted(() => {
  if (folderInput.value) folderInput.value.webkitdirectory = true
})

const ready = computed(() => entries.value.filter((e) => e.state === 'ready'))
const totalSize = computed(() => entries.value.reduce((sum, e) => sum + e.size, 0))

function add(list: { file: File; path: string }[]) {
  // 隐藏文件（.DS_Store 等）和 Office 锁文件直接忽略
  list = list.filter(({ file }) => !file.name.startsWith('.') && !file.name.startsWith('~$'))
  if (!list.length) return
  const byPath = new Map(entries.value.map((e) => [e.path, e]))
  for (const { file, path } of list) {
    const type = (path.split('.').pop() ?? '').toLowerCase()
    byPath.set(path, { path, size: file.size, type, state: 'uploading' })
  }
  entries.value = [...byPath.values()]
  for (let i = 0; i < list.length; i += BATCH) {
    const batch = list.slice(i, i + BATCH)
    queue = queue.then(() => send(batch))   // 按顺序上传，第一批拿到的 upload_id 给后面几批用
  }
}

async function send(batch: { file: File; path: string }[]) {
  // 一批的结果一次写回：defineModel 写入后要等父组件重新渲染才读得到新值，同一轮里连写几次会互相覆盖
  const patches = new Map<string, Partial<Entry>>()
  try {
    const result = await api.upload(batch, uploadId.value || undefined)
    uploadId.value = result.upload_id
    for (const f of result.files) patches.set(f.path, { size: f.size, state: f.supported ? 'ready' : 'unsupported' })
  } catch (err) {
    for (const { path } of batch) patches.set(path, { state: 'error', message: (err as Error).message })
  }
  entries.value = entries.value.map((e) => (patches.has(e.path) ? { ...e, ...patches.get(e.path) } : e))
}

async function remove(entry: Entry) {
  entries.value = entries.value.filter((e) => e.path !== entry.path)
  if (uploadId.value && entry.state !== 'error') {
    try {
      await api.unstage(uploadId.value, entry.path)
    } catch {
      /* 暂存区一天后自动清理，删除失败不影响使用 */
    }
  }
}

function onPick(event: Event, folder: boolean) {
  const input = event.target as HTMLInputElement
  const files = Array.from(input.files ?? [])
  add(files.map((file) => ({ file, path: folder && file.webkitRelativePath ? file.webkitRelativePath : file.name })))
  input.value = ''
}

async function onDrop(event: DragEvent) {
  dragging.value = false
  // webkitGetAsEntry 必须在事件回调里同步取出，await 之后 DataTransfer 就失效了
  const items = Array.from(event.dataTransfer?.items ?? [])
    .map((item) => item.webkitGetAsEntry?.())
    .filter((entry): entry is FileSystemEntry => !!entry)
  const plain = Array.from(event.dataTransfer?.files ?? [])
  if (!items.length) {
    add(plain.map((file) => ({ file, path: file.name })))
    return
  }
  const out: { file: File; path: string }[] = []
  for (const entry of items) await walk(entry, out)
  add(out)
}

async function walk(entry: FileSystemEntry, out: { file: File; path: string }[]) {
  if (entry.isFile) {
    const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject))
    out.push({ file, path: entry.fullPath.replace(/^\//, '') })
  } else if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader()
    let batch: FileSystemEntry[]
    do {   // readEntries 一次最多返回 100 个，要读到空为止
      batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject))
      for (const child of batch) await walk(child, out)
    } while (batch.length)
  }
}

function split(path: string): [string, string] {
  const i = path.lastIndexOf('/')
  return i < 0 ? ['', path] : [path.slice(0, i + 1), path.slice(i + 1)]
}
</script>

<template>
  <div class="file-drop">
    <div class="zone" :class="{ dragging, compact }" @dragover.prevent="dragging = true"
         @dragleave.prevent="dragging = false" @drop.prevent="onDrop">
      <el-icon class="zone-icon"><UploadFilled /></el-icon>
      <div class="zone-title">拖拽文件或文件夹到这里</div>
      <div class="zone-actions">
        <el-button :icon="Document" @click="fileInput?.click()">选择文件</el-button>
        <el-button :icon="FolderOpened" @click="folderInput?.click()">选择文件夹</el-button>
      </div>
      <div class="hint">PDF · Word · PPT · Excel · CSV · Markdown · 网页 · 图片</div>
      <input ref="fileInput" type="file" multiple hidden @change="onPick($event, false)" />
      <input ref="folderInput" type="file" multiple hidden @change="onPick($event, true)" />
    </div>

    <div v-if="entries.length" class="list-head">
      <span>文件（{{ entries.length }}）<span class="muted"> · 可解析 {{ ready.length }}</span></span>
      <span class="muted">共 {{ bytes(totalSize) }}</span>
    </div>
    <div v-if="entries.length" class="list" :class="{ compact }">
      <TransitionGroup name="list">
        <div v-for="entry in entries" :key="entry.path" class="row" :class="{ dim: entry.state === 'unsupported' }">
          <span class="type-badge" :class="`type-${typeGroup(entry.type)}`">{{ entry.type || '?' }}</span>
          <span class="path" :title="entry.path">
            <span class="dir">{{ split(entry.path)[0] }}</span>{{ split(entry.path)[1] }}
          </span>
          <span class="size">{{ bytes(entry.size) }}</span>
          <span class="state">
            <el-icon v-if="entry.state === 'uploading'" class="spin"><Loading /></el-icon>
            <el-icon v-else-if="entry.state === 'ready'" class="ok"><CircleCheckFilled /></el-icon>
            <el-tag v-else-if="entry.state === 'unsupported'" size="small" type="info">不支持</el-tag>
            <el-tooltip v-else :content="entry.message" placement="top">
              <el-icon class="err"><WarningFilled /></el-icon>
            </el-tooltip>
          </span>
          <el-button text circle size="small" :icon="Close" :disabled="entry.state === 'uploading'"
                     @click="remove(entry)" />
        </div>
      </TransitionGroup>
    </div>
  </div>
</template>

<style scoped>
.zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 34px 20px 26px;
  border: 1.5px dashed var(--border-strong);
  border-radius: var(--radius);
  background: var(--surface-2);
  text-align: center;
  transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s var(--ease);
}
.zone.compact { padding: 22px 16px 18px; }
.zone.dragging {
  border-color: var(--primary);
  background: color-mix(in srgb, var(--primary) 7%, var(--surface));
  transform: scale(1.01);
}
.zone-icon { font-size: 34px; color: var(--text-3); transition: color 0.2s ease, transform 0.25s var(--ease); }
.zone.dragging .zone-icon { color: var(--primary); transform: translateY(-3px); }
.zone-title { font-weight: 500; }
.zone-actions { display: flex; gap: 8px; }
.list-head {
  display: flex;
  justify-content: space-between;
  margin: 16px 2px 6px;
  font-size: 13px;
  font-weight: 500;
}
.list {
  position: relative;
  max-height: 380px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
}
.list.compact { max-height: 220px; }
.row {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto 26px 28px;
  align-items: center;
  gap: 10px;
  padding: 6px 8px 6px 10px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}
.row:last-child { border-bottom: none; }
.row.dim { opacity: 0.55; }
.path { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }
.dir { color: var(--text-3); }
.size { font-size: 12px; color: var(--text-3); font-variant-numeric: tabular-nums; }
.state { display: inline-flex; justify-content: center; }
.spin { animation: spin 1s linear infinite; color: var(--primary); }
.ok { color: var(--success); }
.err { color: var(--danger); }
</style>
