<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { MagicStick } from '@element-plus/icons-vue'
import { api, type Recommendation, type Settings } from '../api'
import { clone, settingsError } from '../format'
import { loadEnv, refreshKbs, store } from '../store'
import FileDrop, { type Entry } from '../components/FileDrop.vue'
import SettingsForm from '../components/SettingsForm.vue'

const router = useRouter()
const name = ref('')
const uploadId = ref('')
const entries = ref<Entry[]>([])
const settings = ref<Settings | null>(null)
const recommendation = ref<Recommendation | null>(null)
const recommendedFor = ref('')
const flashKey = ref(0)
const recommending = ref(false)
const submitting = ref(false)

loadEnv().then((env) => {
  if (env && !settings.value) settings.value = clone(env.defaults)
})

const readyPaths = computed(() => entries.value.filter((e) => e.state === 'ready').map((e) => e.path).join('\n'))
const uploading = computed(() => entries.value.some((e) => e.state === 'uploading'))
const stale = computed(() => !!recommendation.value && recommendedFor.value !== readyPaths.value)

const blocker = computed(() => {
  if (!store.online) return '后端未连接'
  if (uploading.value) return '文件上传中…'
  if (!readyPaths.value) return '先添加至少一个可解析的文件'
  return settings.value ? settingsError(settings.value) : '正在读取环境信息…'
})

async function recommend() {
  if (!uploadId.value) return
  recommending.value = true
  try {
    const result = await api.recommend(uploadId.value)
    recommendation.value = result
    recommendedFor.value = readyPaths.value
    settings.value = clone(result.settings)
    flashKey.value++
  } catch (err) {
    ElMessage.error((err as Error).message)
  } finally {
    recommending.value = false
  }
}

// 手动改了参数，推荐理由就不再准确，改动的字段不再显示"推荐"
watch(
  settings,
  (now) => {
    const rec = recommendation.value
    if (!rec || !now) return
    const reasons = rec.reasons
    const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)
    const checks: Record<string, () => boolean> = {
      chunker: () => now.chunker.type === rec.settings.chunker.type,
      chunk_size: () => now.chunker.chunk_size === rec.settings.chunker.chunk_size,
      parent_size: () => now.chunker.parent_size === rec.settings.chunker.parent_size,
      overlap_sentences: () => now.chunker.overlap_sentences === rec.settings.chunker.overlap_sentences,
      encoder: () => same(now.encoder, rec.settings.encoder),
      index: () => now.index.type === rec.settings.index.type,
      pdf_parser: () => same(now.parsers['.pdf'], rec.settings.parsers['.pdf']),
      docx_parser: () => same(now.parsers['.docx'], rec.settings.parsers['.docx']),
      ocr: () => now.ocr === rec.settings.ocr,
    }
    for (const key of Object.keys(reasons)) if (checks[key] && !checks[key]()) delete reasons[key]
  },
  { deep: true },
)

async function submit() {
  if (blocker.value || !settings.value) return
  submitting.value = true
  try {
    const { kb, job } = await api.createKb(name.value.trim() || '未命名知识库', uploadId.value, settings.value)
    ElMessage.success('已开始建库')
    await refreshKbs()
    router.push({ path: `/kb/${kb.id}`, query: { job: job.id } })
  } catch (err) {
    ElMessage.error((err as Error).message)
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1>新建知识库</h1>
        <div class="sub">选好文件和参数后开始建库。建库在后台进行，可以随时离开这个页面</div>
      </div>
    </div>

    <el-input v-model="name" size="large" maxlength="40" placeholder="知识库名称，例如：产品手册" class="name-input" />

    <div class="layout">
      <section class="card rise" style="--i: 1">
        <div class="card-head"><h3>文件</h3><span class="hint">保留文件夹结构，同名文件在不同目录下不会冲突</span></div>
        <div class="card-body">
          <FileDrop v-model:upload-id="uploadId" v-model:entries="entries" />
        </div>
      </section>

      <section class="card params rise" style="--i: 2">
        <div class="card-head">
          <h3>参数</h3>
          <el-tooltip content="扫描上传的文件，按规则推荐每一项参数并说明理由" placement="top">
            <el-button size="small" :icon="MagicStick" :loading="recommending" :disabled="!readyPaths || uploading"
                       @click="recommend">一键推荐</el-button>
          </el-tooltip>
        </div>
        <div class="card-body">
          <Transition name="slide">
            <div v-if="recommendation" class="scan">
              <div class="scan-head">
                <span>扫描结果</span>
                <el-button v-if="stale" link type="primary" size="small" @click="recommend">文件有变化，重新推荐</el-button>
              </div>
              <div v-for="(line, i) in recommendation.summary" :key="i" class="scan-line" :class="{ first: i === 0 }">
                {{ line }}
              </div>
              <el-alert v-for="w in recommendation.warnings" :key="w" :title="w" type="warning" :closable="false"
                        show-icon class="warn" />
            </div>
          </Transition>
          <SettingsForm v-if="settings && store.env" v-model="settings" :env="store.env"
                        :reasons="recommendation?.reasons" :flash-key="flashKey" />
          <el-skeleton v-else :rows="8" animated />
        </div>
        <div class="footer">
          <el-button type="primary" size="large" class="submit" :loading="submitting" :disabled="!!blocker"
                     @click="submit">开始建库</el-button>
          <div v-if="blocker" class="hint blocker">{{ blocker }}</div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.name-input { max-width: 520px; margin-bottom: 18px; }
.name-input :deep(.el-input__inner) { font-size: 15px; }
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 420px;
  gap: 18px;
  align-items: start;
}
.params { position: sticky; top: 20px; }
.params .card-body { max-height: calc(100vh - 260px); overflow-y: auto; padding-top: 10px; }
.scan {
  margin: 4px 0 10px;
  padding: 12px 14px;
  border-radius: 8px;
  background: var(--surface-2);
  border: 1px solid var(--border);
}
.scan-head { display: flex; justify-content: space-between; align-items: center; font-size: 12px; color: var(--text-3); margin-bottom: 4px; }
.scan-line { font-size: 13px; color: var(--text-2); }
.scan-line.first { color: var(--text); font-weight: 500; }
.warn { margin-top: 8px; }
.footer { padding: 14px 18px 16px; border-top: 1px solid var(--border); }
.submit { width: 100%; }
.blocker { text-align: center; margin-top: 6px; }
@media (max-width: 1080px) {
  .layout { grid-template-columns: 1fr; }
  .params { position: static; }
  .params .card-body { max-height: none; }
}
</style>
