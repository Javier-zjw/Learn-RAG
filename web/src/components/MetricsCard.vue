<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { api, type Latency, type Metrics } from '../api'
import { timeAgo } from '../format'

// 运行监控：线上查询（learn-rag chat、以后的提问页）的延迟、疑似空召回、错误，和建库的耗时
const props = defineProps<{ kbId: string; refreshKey: number }>()

const days = ref(7)
const data = ref<Metrics | null>(null)
const error = ref('')

async function load() {
  try {
    data.value = await api.metrics(props.kbId, days.value)
    error.value = ''
  } catch (err) {
    error.value = (err as Error).message
  }
}
watch(() => [props.kbId, props.refreshKey, days.value], load, { immediate: true })

const STAGES: Record<string, string> = {
  condense: '问题补全', retrieve: '召回', rerank: '精排', expand: '父块展开', generate: '生成', total: '合计',
  parse: '解析', index: '切分与向量化',
}
const KINDS: Record<string, string> = { build: '建库', append: '追加', rebuild: '重建', remove: '删除文档' }
const JOB_STATES: Record<string, string> = { queued: '排队中', running: '执行中', done: '完成', failed: '失败' }

const q = computed(() => data.value?.queries ?? null)
const hasQueries = computed(() => !!q.value && q.value.queries > 0)
const stages = computed(() => {
  const latency = q.value?.latency ?? {}
  // 合计放最后，其余按管线顺序
  const order = ['condense', 'retrieve', 'rerank', 'expand', 'generate', 'total']
  return order.filter((k) => latency[k]).map((k) => ({ name: STAGES[k], ...latency[k] }))
})
const buildStages = computed(() => {
  const stages = data.value?.build.stages ?? {}
  return ['parse', 'index'].filter((k) => stages[k]).map((k) => ({ name: STAGES[k], ...stages[k] }))
})

function ms(seconds: number): string {
  if (seconds > 0 && seconds < 0.0005) return '<1 ms'
  if (seconds >= 1) return `${seconds.toFixed(seconds >= 10 ? 0 : 2)} s`
  return `${Math.round(seconds * 1000)} ms`
}
function pct(rate: number): string {
  return `${(rate * 100).toFixed(rate && rate < 0.1 ? 1 : 0)}%`
}
// 状态只在"值得注意"时提示，并且总是配文字，不靠颜色单独表达
const emptyHigh = computed(() => hasQueries.value && q.value!.empty_rate >= 0.1)
const errorHigh = computed(() => hasQueries.value && q.value!.error_rate >= 0.05)
const p95 = computed<Latency | null>(() => q.value?.latency.total ?? null)
</script>

<template>
  <section class="card metrics">
    <div class="card-head">
      <h3>运行监控</h3>
      <el-radio-group v-model="days" size="small">
        <el-radio-button :value="1">1 天</el-radio-button>
        <el-radio-button :value="7">7 天</el-radio-button>
        <el-radio-button :value="30">30 天</el-radio-button>
      </el-radio-group>
    </div>
    <div class="card-body">
      <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon />
      <template v-else-if="data">
        <div class="tiles">
          <div class="tile">
            <span>查询</span>
            <b>{{ q!.queries }}</b>
            <small>次</small>
          </div>
          <div class="tile">
            <span>P95 延迟</span>
            <b>{{ p95 ? ms(p95.p95) : '—' }}</b>
            <small>{{ p95 ? `P50 ${ms(p95.p50)}` : '还没有查询' }}</small>
          </div>
          <div class="tile">
            <span>疑似空召回</span>
            <b>{{ hasQueries ? pct(q!.empty_rate) : '—' }}</b>
            <small :class="{ warn: emptyHigh }">{{ emptyHigh ? '偏高：看看下面的问题' : '召回的内容和问题没有共同的词' }}</small>
          </div>
          <div class="tile">
            <span>错误率</span>
            <b>{{ hasQueries ? pct(q!.error_rate) : '—' }}</b>
            <small :class="{ bad: errorHigh }">{{ errorHigh ? `偏高：${q!.errors} 次出错` : `${q!.errors} 次出错` }}</small>
          </div>
        </div>

        <div v-if="!hasQueries" class="empty-hint">
          最近 {{ days }} 天没有线上查询。用 <code>learn-rag chat --kb {{ kbId }}</code> 提问后，这里会显示延迟和疑似空召回；
          评测和调参不计入。
        </div>
        <div v-else class="columns">
          <div>
            <div class="sub">各阶段耗时</div>
            <table class="grid">
              <thead><tr><th>阶段</th><th>P50</th><th>P95</th><th>最大</th></tr></thead>
              <tbody>
                <tr v-for="s in stages" :key="s.name" :class="{ total: s.name === '合计' }">
                  <td>{{ s.name }}</td><td>{{ ms(s.p50) }}</td><td>{{ ms(s.p95) }}</td><td>{{ ms(s.max) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div>
            <div class="sub">最近疑似空召回的问题 <span class="muted">· 补文档、补评测集的素材</span></div>
            <ul v-if="q!.recent_empty.length" class="questions">
              <li v-for="e in q!.recent_empty" :key="e.ts">
                <span class="q">{{ e.question }}</span>
                <span class="muted">{{ timeAgo(e.ts) }}</span>
              </li>
            </ul>
            <div v-else class="muted">没有</div>
            <template v-if="q!.recent_errors.length">
              <div class="sub">最近出错的问题</div>
              <ul class="questions">
                <li v-for="e in q!.recent_errors" :key="e.ts">
                  <span class="q">{{ e.question }}<span class="err">{{ e.error }}</span></span>
                  <span class="muted">{{ timeAgo(e.ts) }}</span>
                </li>
              </ul>
            </template>
          </div>
        </div>

        <div class="columns build">
          <div>
            <div class="sub">建库耗时（每个文件，{{ data.build.files }} 个文件）</div>
            <table v-if="buildStages.length" class="grid">
              <thead><tr><th>阶段</th><th>P50</th><th>P95</th><th>合计</th></tr></thead>
              <tbody>
                <tr v-for="s in buildStages" :key="s.name">
                  <td>{{ s.name }}</td><td>{{ ms(s.p50) }}</td><td>{{ ms(s.p95) }}</td><td>{{ ms(s.total) }}</td>
                </tr>
              </tbody>
            </table>
            <div v-else class="muted">还没有记录（升级前建的库重建一次后才有）</div>
          </div>
          <div>
            <div class="sub">最近的任务</div>
            <table v-if="data.build.jobs.length" class="grid">
              <thead><tr><th>任务</th><th>状态</th><th>排队</th><th>执行</th><th>次数</th><th>失败</th></tr></thead>
              <tbody>
                <tr v-for="j in data.build.jobs" :key="j.id">
                  <td>{{ KINDS[j.kind] ?? j.kind }} <span class="muted">{{ timeAgo(j.created_at) }}</span></td>
                  <td>{{ JOB_STATES[j.status] ?? j.status }}</td>
                  <td>{{ j.queued_seconds != null ? ms(j.queued_seconds) : '—' }}</td>
                  <td>{{ j.run_seconds != null ? ms(j.run_seconds) : '—' }}</td>
                  <td>{{ j.attempts }}</td>
                  <td :class="{ bad: j.failed }">{{ j.failed ? `${j.failed} 个文件` : '0' }}</td>
                </tr>
              </tbody>
            </table>
            <div v-else class="muted">没有任务记录</div>
          </div>
        </div>
      </template>
      <el-skeleton v-else :rows="4" animated />
    </div>
  </section>
</template>

<style scoped>
.metrics { margin-top: 18px; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
.tile {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface-2);
}
.tile span { font-size: 12px; color: var(--text-3); }
.tile b { font-size: 22px; font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; line-height: 1.3; }
.tile small { font-size: 12px; color: var(--text-3); }
.tile small.warn { color: var(--warning); font-weight: 500; }
.tile small.bad, td.bad { color: var(--danger); font-weight: 500; }
.columns { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.columns.build { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--border); }
.sub { font-size: 12px; font-weight: 600; color: var(--text-3); margin: 0 0 8px; }
.sub:not(:first-child) { margin-top: 14px; }
.grid { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
.grid th { text-align: left; font-weight: 500; color: var(--text-3); padding: 4px 8px 6px 0; border-bottom: 1px solid var(--border); }
.grid td { padding: 6px 8px 6px 0; color: var(--text-2); border-bottom: 1px solid var(--border); }
.grid td:first-child { color: var(--text); }
.grid tr.total td { font-weight: 600; color: var(--text); }
.questions { list-style: none; margin: 0; padding: 0; font-size: 13px; }
.questions li { display: flex; justify-content: space-between; gap: 12px; padding: 5px 0; border-bottom: 1px solid var(--border); }
.questions .q { color: var(--text); min-width: 0; overflow-wrap: anywhere; }
.questions .err { display: block; font-size: 12px; color: var(--text-3); }
.muted { color: var(--text-3); font-size: 12px; font-weight: 400; white-space: nowrap; }
.empty-hint { font-size: 13px; color: var(--text-2); line-height: 1.6; }
.empty-hint code { font-family: var(--mono); font-size: 12px; background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
@media (max-width: 900px) {
  .tiles { grid-template-columns: repeat(2, 1fr); }
  .columns { grid-template-columns: 1fr; }
}
</style>
