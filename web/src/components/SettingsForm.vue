<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { CircleCheckFilled } from '@element-plus/icons-vue'
import type { EnvReport, Settings } from '../api'
import { settingsError } from '../format'
import FormField from './FormField.vue'

const settings = defineModel<Settings>({ required: true })
const props = defineProps<{ env: EnvReport; reasons?: Record<string, string>; flashKey?: number }>()

const reasons = computed(() => props.reasons ?? {})
const advancedOpen = ref<string[]>([])
const flashing = ref(new Set<string>())
const ADVANCED = ['index', 'pdf_parser', 'docx_parser', 'ocr']

// 推荐值填入后：被推荐的字段闪一下；推荐涉及高级参数时自动展开
watch(
  () => props.flashKey,
  () => {
    const keys = Object.keys(reasons.value)
    flashing.value = new Set(keys)
    if (keys.some((k) => ADVANCED.includes(k))) advancedOpen.value = ['advanced']
    window.setTimeout(() => (flashing.value = new Set()), 1600)
  },
)

// 换切分器时补上它需要的参数
watch(
  () => settings.value.chunker.type,
  (type) => {
    const c = settings.value.chunker
    if (type === 'structure') {
      c.parent_size ??= c.chunk_size * 4
      c.overlap_sentences ??= 0
    } else {
      c.chunk_overlap ??= Math.round(c.chunk_size / 5)
    }
  },
)

const chunkError = computed(() => settingsError(settings.value))

function pickEncoder(type: string, model: string) {
  settings.value.encoder = { type, model }
}

const parserKey = (chain: string[]) => chain.join('>')
function parserModel(suffix: string) {
  return computed({
    get: () => parserKey(settings.value.parsers[suffix] ?? []),
    set: (key: string) => (settings.value.parsers[suffix] = key.split('>')),
  })
}
const pdfParser = parserModel('.pdf')
const docxParser = parserModel('.docx')
const chromaAvailable = computed(() => props.env.packages.some((p) => p.name.startsWith('Chroma') && p.installed))
</script>

<template>
  <div class="settings">
    <div class="section-label">切分</div>
    <FormField label="切分器" :reason="reasons.chunker" :flash="flashing.has('chunker')">
      <el-select v-model="settings.chunker.type">
        <el-option value="structure" label="结构感知父子分块（推荐）" />
        <el-option value="recursive" label="递归字符切分" />
        <el-option value="fixed" label="定长滑窗" />
      </el-select>
    </FormField>

    <template v-if="settings.chunker.type === 'structure'">
      <FormField label="子块大小" unit="token" :reason="reasons.chunk_size" :flash="flashing.has('chunk_size')"
                 hint="子块进向量索引和 BM25，负责召回；中文约一字一个 token">
        <el-slider v-model="settings.chunker.chunk_size" :min="100" :max="1000" :step="50" show-input
                   :show-input-controls="false" size="small" />
      </FormField>
      <FormField label="父块大小" unit="token" :reason="reasons.parent_size" :flash="flashing.has('parent_size')"
                 :error="chunkError" hint="命中子块后换成父块交给大模型；0 表示不生成父块">
        <el-input-number v-model="settings.chunker.parent_size" :min="0" :max="8000" :step="100"
                         controls-position="right" />
      </FormField>
      <FormField label="句子重叠" unit="句" :reason="reasons.overlap_sentences"
                 :flash="flashing.has('overlap_sentences')" hint="长段落切成几块时，相邻子块重复的句子数">
        <el-input-number v-model="settings.chunker.overlap_sentences" :min="0" :max="3" controls-position="right" />
      </FormField>
    </template>
    <template v-else>
      <FormField label="块大小" unit="字符" :reason="reasons.chunk_size" :flash="flashing.has('chunk_size')">
        <el-slider v-model="settings.chunker.chunk_size" :min="100" :max="2000" :step="50" show-input
                   :show-input-controls="false" size="small" />
      </FormField>
      <FormField label="重叠" unit="字符" :error="chunkError" hint="相邻块重复的字符数">
        <el-input-number v-model="settings.chunker.chunk_overlap" :min="0" :max="1000" :step="10"
                         controls-position="right" />
      </FormField>
    </template>

    <div class="section-label">向量化</div>
    <FormField label="Embedding 模型" :reason="reasons.encoder" :flash="flashing.has('encoder')">
      <div class="encoders">
        <el-tooltip v-for="option in env.encoders" :key="option.type" :content="option.note"
                    :disabled="option.available" placement="left">
          <button type="button" class="encoder" :disabled="!option.available"
                  :class="{ selected: settings.encoder.type === option.type }"
                  @click="pickEncoder(option.type, option.model)">
            <span class="enc-main">
              <span class="enc-label">{{ option.label }}</span>
              <span class="enc-note">{{ option.note }}</span>
            </span>
            <el-icon class="enc-check"><CircleCheckFilled /></el-icon>
          </button>
        </el-tooltip>
      </div>
      <Transition name="slide">
        <el-input v-if="settings.encoder.type === 'openai_compat'" v-model="settings.encoder.model" size="small"
                  class="model-input">
          <template #prepend>模型名</template>
        </el-input>
      </Transition>
    </FormField>

    <el-collapse v-model="advancedOpen" class="advanced">
      <el-collapse-item name="advanced" title="高级：向量库、解析器、OCR">
        <FormField label="向量库" :reason="reasons.index" :flash="flashing.has('index')">
          <el-radio-group v-model="settings.index.type">
            <el-radio-button value="chroma" :disabled="!chromaAvailable">Chroma（持久化）</el-radio-button>
            <el-radio-button value="flat">Flat（内存）</el-radio-button>
          </el-radio-group>
        </FormField>
        <template v-if="settings.index.type === 'chroma'">
          <FormField label="距离度量">
            <el-select v-model="settings.index.space">
              <el-option value="cosine" label="cosine 余弦" />
              <el-option value="ip" label="ip 内积" />
              <el-option value="l2" label="l2 欧氏距离" />
            </el-select>
          </FormField>
          <div class="hnsw">
            <FormField label="ef_construction" hint="建图质量">
              <el-input-number v-model="settings.index.ef_construction" :min="16" :max="1000" :step="20"
                               controls-position="right" />
            </FormField>
            <FormField label="max_neighbors" hint="HNSW 的 M">
              <el-input-number v-model="settings.index.max_neighbors" :min="4" :max="128" :step="4"
                               controls-position="right" />
            </FormField>
            <FormField label="ef_search" hint="查询宽度">
              <el-input-number v-model="settings.index.ef_search" :min="10" :max="1000" :step="10"
                               controls-position="right" />
            </FormField>
          </div>
        </template>
        <FormField label="PDF 解析器" :reason="reasons.pdf_parser" :flash="flashing.has('pdf_parser')">
          <el-select v-model="pdfParser">
            <el-option v-for="c in env.parsers['.pdf']" :key="parserKey(c.value)" :value="parserKey(c.value)"
                       :label="c.label" :disabled="!c.available">
              <span>{{ c.label }}</span><span v-if="c.note" class="opt-note">{{ c.note }}</span>
            </el-option>
          </el-select>
        </FormField>
        <FormField label="Word 解析器" :reason="reasons.docx_parser" :flash="flashing.has('docx_parser')">
          <el-select v-model="docxParser">
            <el-option v-for="c in env.parsers['.docx']" :key="parserKey(c.value)" :value="parserKey(c.value)"
                       :label="c.label" :disabled="!c.available">
              <span>{{ c.label }}</span><span v-if="c.note" class="opt-note">{{ c.note }}</span>
            </el-option>
          </el-select>
        </FormField>
        <FormField label="OCR（扫描页和图片）" :reason="reasons.ocr" :flash="flashing.has('ocr')"
                   :hint="env.ocr ? '使用 .env 中配置的视觉模型服务' : '未配置 OCR 服务（.env 中的 OCR_BASE_URL）'">
          <el-switch v-model="settings.ocr" :disabled="!env.ocr" />
        </FormField>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<style scoped>
.settings { display: flex; flex-direction: column; }
.section-label {
  margin: 6px 0 2px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--text-3);
}
.section-label:not(:first-child) { margin-top: 14px; }
.el-select, .el-input-number { width: 100%; }
:deep(.el-slider__runway.show-input) { margin-right: 16px; }
:deep(.el-slider__input) { width: 76px; }
.encoders { display: flex; flex-direction: column; gap: 8px; }
.encoder {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-strong);
  border-radius: 8px;
  background: var(--surface);
  color: var(--text);
  font: inherit;
  text-align: left;
  cursor: pointer;
  transition: border-color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
}
.encoder:hover:not(:disabled) { border-color: color-mix(in srgb, var(--primary) 55%, var(--border)); }
.encoder.selected {
  border-color: var(--primary);
  background: color-mix(in srgb, var(--primary) 6%, var(--surface));
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 12%, transparent);
}
.encoder:disabled { cursor: not-allowed; opacity: 0.5; background: var(--surface-2); }
.enc-main { display: flex; flex-direction: column; min-width: 0; flex: 1; }
.enc-label { font-weight: 500; font-size: 13px; }
.enc-note { font-size: 12px; color: var(--text-3); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.enc-check { color: var(--primary); font-size: 16px; opacity: 0; transform: scale(0.6); transition: all 0.2s var(--ease); }
.encoder.selected .enc-check { opacity: 1; transform: scale(1); }
.model-input { margin-top: 2px; }
.advanced { margin-top: 12px; border-top: 1px solid var(--border); border-bottom: none; }
.advanced :deep(.el-collapse-item__header) { font-weight: 500; color: var(--text-2); background: transparent; border-bottom: none; }
.advanced :deep(.el-collapse-item__wrap) { background: transparent; border-bottom: none; }
.advanced :deep(.el-collapse-item__content) { padding-bottom: 4px; }
.hnsw { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0 10px; }
.opt-note { margin-left: 8px; font-size: 12px; color: var(--text-3); }
</style>
