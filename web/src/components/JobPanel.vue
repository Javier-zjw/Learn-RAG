<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { ArrowDown, CircleCheckFilled, CircleCloseFilled, Loading, Close } from '@element-plus/icons-vue'
import type { FileProgress, Job } from '../api'
import { STATUS } from '../format'

const props = defineProps<{ job: Job }>()
defineEmits<{ (e: 'close'): void }>()

const total = computed(() => props.job.files.length)
const running = computed(() => props.job.status === 'queued' || props.job.status === 'running')
const percent = computed(() => (total.value ? Math.round((props.job.processed / total.value) * 100) : 100))
const failed = computed(() => props.job.counts.failed)
const expanded = ref(true)
// 跑完且没有失败时自动收起文件列表，有失败时保持展开方便看原因
watch(running, (now, before) => {
  if (before && !now && !failed.value) expanded.value = false
})

const KIND = { build: '建库', append: '追加文件', rebuild: '重建' } as const
const title = computed(() => {
  const kind = KIND[props.job.kind]
  if (props.job.status === 'queued') return `${kind}排队中`
  if (running.value) return `正在${kind}`
  if (props.job.status === 'failed') return `${kind}失败`
  return failed.value ? `${kind}完成，${failed.value} 个文件失败` : `${kind}完成`
})
const progressStatus = computed(() => {
  if (running.value) return undefined
  if (props.job.status === 'failed') return 'exception'
  return failed.value ? 'warning' : 'success'
})

// 每个文件的三步：解析 → 切分与向量化 → 完成
const STEPS = ['解析', '切分与向量化', '入库']
function stepIndex(f: FileProgress): number {
  return { waiting: -1, parsing: 0, embedding: 1, done: 3, skipped: 3, failed: -2 }[f.status]
}
</script>

<template>
  <div class="card job" :class="{ running }">
    <div class="job-head">
      <div class="job-title">
        <el-icon v-if="running" class="spin"><Loading /></el-icon>
        <el-icon v-else-if="job.status === 'failed'" class="bad"><CircleCloseFilled /></el-icon>
        <el-icon v-else :class="failed ? 'warn' : 'good'"><CircleCheckFilled /></el-icon>
        <span>{{ title }}</span>
        <span class="count">{{ job.processed }} / {{ total }}</span>
      </div>
      <div class="job-actions">
        <el-button text size="small" @click="expanded = !expanded">
          {{ expanded ? '收起' : '展开' }}
          <el-icon class="arrow" :class="{ up: expanded }"><ArrowDown /></el-icon>
        </el-button>
        <el-button v-if="!running" text circle size="small" :icon="Close" @click="$emit('close')" />
      </div>
    </div>
    <el-progress :percentage="percent" :status="progressStatus" :striped="running" :striped-flow="running"
                 :duration="12" :stroke-width="8" :show-text="false" class="bar" />
    <div v-if="!running" class="tally">
      <span>完成 {{ job.counts.done }}</span>
      <span v-if="job.counts.skipped">未变化 {{ job.counts.skipped }}</span>
      <span v-if="failed" class="bad-text">失败 {{ failed }}</span>
      <template v-if="job.verify">
        <span v-if="job.verify.ok" class="good-text">
          完整性检查通过（{{ job.verify.chunks }} 个子块、{{ job.verify.parents }} 个父块<template
            v-if="job.verify.sampled">，抽样核对 {{ job.verify.sampled }} 个向量</template>）
        </span>
        <span v-else class="bad-text">完整性检查发现 {{ job.verify.problems.length }} 个问题</span>
      </template>
    </div>
    <Transition name="slide">
      <el-alert v-if="job.notice" :title="job.notice" :type="running ? 'info' : 'warning'" :closable="false"
                show-icon class="alert" />
    </Transition>
    <el-alert v-if="job.error" :title="job.error" type="error" :closable="false" show-icon class="alert" />
    <el-alert v-if="job.verify && !job.verify.ok" type="warning" :closable="false" class="alert">
      <div v-for="p in job.verify.problems" :key="p">{{ p }}</div>
    </el-alert>

    <el-collapse-transition>
      <div v-show="expanded" class="files">
        <div v-for="f in job.files" :key="f.path" class="file" :class="f.status">
          <div class="file-main">
            <span class="file-path" :title="f.path">{{ f.path }}</span>
            <span class="steps">
              <span v-for="(step, i) in STEPS" :key="step" class="step"
                    :class="{ done: stepIndex(f) > i, now: stepIndex(f) === i, fail: f.status === 'failed' }">
                {{ step }}
              </span>
            </span>
            <span class="file-result">
              <template v-if="f.status === 'done' || f.status === 'skipped'">{{ f.chunks }} 子块 · {{ f.parents }} 父块</template>
            </span>
            <el-tag size="small" :type="STATUS[f.status].type" effect="light" class="tag">{{ STATUS[f.status].label }}</el-tag>
          </div>
          <div v-if="f.message && f.status !== 'skipped'" class="file-msg">{{ f.message }}</div>
        </div>
      </div>
    </el-collapse-transition>
  </div>
</template>

<style scoped>
.job { padding: 14px 18px 10px; margin-bottom: 18px; }
.job.running { border-color: color-mix(in srgb, var(--primary) 30%, var(--border)); }
.job-head { display: flex; align-items: center; justify-content: space-between; }
.job-title { display: flex; align-items: center; gap: 8px; font-weight: 600; }
.count { font-weight: 400; color: var(--text-3); font-variant-numeric: tabular-nums; }
.job-actions { display: flex; align-items: center; }
.arrow { margin-left: 2px; transition: transform 0.2s var(--ease); }
.arrow.up { transform: rotate(180deg); }
.bar { margin: 10px 0 6px; }
.tally { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12.5px; color: var(--text-2); }
.alert { margin-top: 8px; }
.files { margin-top: 8px; max-height: 320px; overflow-y: auto; border-top: 1px solid var(--border); }
.file { padding: 7px 2px; border-bottom: 1px solid var(--border); transition: background 0.3s ease; }
.file:last-child { border-bottom: none; }
.file.parsing, .file.embedding { background: color-mix(in srgb, var(--primary) 4%, transparent); }
.file-main { display: grid; grid-template-columns: minmax(0, 1fr) auto 120px 70px; align-items: center; gap: 12px; }
.file-path { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }
.steps { display: flex; gap: 4px; }
.step {
  padding: 1px 8px;
  border-radius: 10px;
  font-size: 11.5px;
  color: var(--text-3);
  background: var(--surface-2);
  transition: background 0.3s ease, color 0.3s ease;
}
.step.done { color: var(--success); background: color-mix(in srgb, var(--success) 10%, transparent); }
.step.now { color: var(--primary); background: color-mix(in srgb, var(--primary) 12%, transparent); animation: pulse 1.4s ease-in-out infinite; }
.step.fail { color: var(--text-3); background: var(--surface-2); }
@keyframes pulse { 50% { opacity: 0.55; } }
.file-result { font-size: 12px; color: var(--text-3); text-align: right; font-variant-numeric: tabular-nums; }
.tag { justify-self: end; }
.file-msg {
  margin-top: 4px;
  padding: 6px 9px;
  border-radius: 6px;
  font: 12px/1.5 var(--mono);
  white-space: pre-wrap;
  color: var(--danger);
  background: color-mix(in srgb, var(--danger) 7%, transparent);
}
.spin { animation: spin 1s linear infinite; color: var(--primary); }
.good, .good-text { color: var(--success); }
.bad, .bad-text { color: var(--danger); }
.warn { color: var(--warning); }
@media (max-width: 760px) {
  .steps { display: none; }
  .file-main { grid-template-columns: minmax(0, 1fr) auto 70px; }
}
</style>
