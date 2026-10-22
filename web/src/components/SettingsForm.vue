<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { CircleCheckFilled } from '@element-plus/icons-vue'
import type { Channel, EnvReport, Settings } from '../api'
import { SPARSE_ENCODERS, chunkerError, retrievalError } from '../format'
import FormField from './FormField.vue'

const settings = defineModel<Settings>({ required: true })
// part="retrieval" 只显示检索与精排（知识库详情页改精排用，改它不需要重建）
const props = defineProps<{ env: EnvReport; reasons?: Record<string, string>; flashKey?: number; part?: 'all' | 'retrieval' }>()

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

const chunkError = computed(() => chunkerError(settings.value))
const fusionError = computed(() => retrievalError(settings.value))

const CHANNELS: { key: Channel; label: string }[] = [
  { key: 'vector', label: '向量' },
  { key: 'bm25', label: 'BM25' },
  { key: 'sparse', label: '稀疏' },
]
// 查询改写的说明跟着选项走；没配大模型时说明为什么只能选"不改写"
const transformNote = computed(() => {
  const options = props.env.query_transforms
  if (!props.env.llm) return options.find((t) => !t.available)?.note ?? ''
  return options.find((t) => t.type === settings.value.retrieval.query_transform)?.note ?? ''
})
const enrichOn = computed(() => {
  const e = settings.value.enrich
  return e.context || e.describe || e.summaries || e.questions > 0
})
const sparseAvailable = computed(() => SPARSE_ENCODERS.includes(settings.value.encoder.type))

function pickEncoder(type: string, model: string) {
  settings.value.encoder = { type, model }
  // 能生成稀疏向量的模型默认三路等权；换成别的模型时关掉稀疏通道，否则建库会被拒绝
  settings.value.retrieval.weights.sparse = SPARSE_ENCODERS.includes(type) ? 1 : 0
}

function pickReranker(type: string, model: string) {
  settings.value.reranker = { type, model }
  // 真正的精排给它 50 个候选去挑；词重叠精排判别力有限，候选多了反而引入噪声
  settings.value.retrieval.candidate_k = type === 'api' || type === 'cross_encoder' ? 50 : 20
}

const MODEL_INPUT: Record<string, string> = { openai_compat: '模型名', sentence_transformers: '模型名或本地目录',
                                              api: '模型名', cross_encoder: '模型名或本地目录' }

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
    <template v-if="part !== 'retrieval'">
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
        <el-input v-if="MODEL_INPUT[settings.encoder.type]" v-model="settings.encoder.model" size="small"
                  class="model-input">
          <template #prepend>{{ MODEL_INPUT[settings.encoder.type] }}</template>
        </el-input>
      </Transition>
    </FormField>

    <div class="section-label">建库增强</div>
    <FormField label="大模型增强" :reason="reasons.enrich" :flash="flashing.has('enrich')"
               :hint="env.enrich.available ? '生成的文字只用于检索，不改动分块正文；改了要重建' : env.enrich.note">
      <div class="enrich">
        <label class="switch-row">
          <el-switch v-model="settings.enrich.context" :disabled="!env.enrich.available" size="small" />
          <span><b>补背景</b>：每个子块一句它在全文里的背景，补上省略的主语、产品名</span>
        </label>
        <label class="switch-row">
          <el-switch v-model="settings.enrich.describe" :disabled="!env.enrich.available" size="small" />
          <span><b>描述图表</b>：概括表格；{{ env.enrich.vlm ? '图片和图表交给视觉模型描述' : '未配置视觉模型（OCR_MODEL），不描述图片' }}</span>
        </label>
        <label class="switch-row">
          <el-switch v-model="settings.enrich.summaries" :disabled="!env.enrich.available" size="small" />
          <span><b>章节摘要</b>：每个父块一段摘要，单独生成向量，对应“这一节讲什么”一类的提问</span>
        </label>
        <label class="switch-row">
          <el-input-number v-model="settings.enrich.questions" :min="0" :max="5" size="small"
                           controls-position="right" :disabled="!env.enrich.available" class="questions" />
          <span><b>假设问题</b>：每个子块生成几个它能回答的问题，各自一个向量；0 表示不生成</span>
        </label>
      </div>
      <div v-if="env.enrich.available && enrichOn" class="enrich-cost">每个子块调用一次大模型，结果有缓存，重建不会重复调用</div>
    </FormField>
    </template>

    <div class="section-label">检索与精排</div>
    <FormField label="精排模型" :reason="reasons.reranker" :flash="flashing.has('reranker')"
               hint="召回的候选逐个和问题一起打分再排序；只影响查询，改了不用重建">
      <div class="encoders">
        <el-tooltip v-for="option in env.rerankers" :key="option.type" :content="option.note"
                    :disabled="option.available" placement="left">
          <button type="button" class="encoder" :disabled="!option.available"
                  :class="{ selected: settings.reranker.type === option.type }"
                  @click="pickReranker(option.type, option.model)">
            <span class="enc-main">
              <span class="enc-label">{{ option.label }}</span>
              <span class="enc-note">{{ option.note }}</span>
            </span>
            <el-icon class="enc-check"><CircleCheckFilled /></el-icon>
          </button>
        </el-tooltip>
      </div>
      <Transition name="slide">
        <el-input v-if="MODEL_INPUT[settings.reranker.type]" v-model="settings.reranker.model" size="small"
                  class="model-input">
          <template #prepend>{{ MODEL_INPUT[settings.reranker.type] }}</template>
        </el-input>
      </Transition>
    </FormField>
    <div class="pair">
      <FormField label="召回候选数" hint="各路召回融合后交给精排的片段数">
        <el-input-number v-model="settings.retrieval.candidate_k" :min="settings.retrieval.top_k" :max="200" :step="10"
                         controls-position="right" />
      </FormField>
      <FormField label="交给大模型" unit="个片段" hint="精排后取前几个（再展开成父块）">
        <el-input-number v-model="settings.retrieval.top_k" :min="1" :max="20" controls-position="right" />
      </FormField>
    </div>
    <FormField label="融合权重" :reason="reasons.fusion" :flash="flashing.has('fusion')" :error="fusionError"
               :hint="sparseAvailable ? '各路按名次做 RRF 融合，权重越大这一路越说了算；0 表示不用这一路'
                                      : '0 表示不用这一路；稀疏召回需要 embedding 选 BGE-M3（稠密 + 稀疏）'">
      <div class="weights">
        <label v-for="c in CHANNELS" :key="c.key" class="weight">
          <span class="weight-label">{{ c.label }}</span>
          <el-input-number v-model="settings.retrieval.weights[c.key]" :min="0" :max="10" :step="0.5"
                           :precision="1" controls-position="right"
                           :disabled="c.key === 'sparse' && !sparseAvailable" />
        </label>
      </div>
    </FormField>
    <div class="pair">
      <FormField label="RRF 常数 k" hint="越小越看重各路的第一名；60 是常用值">
        <el-input-number v-model="settings.retrieval.rrf_k" :min="1" :max="1000" :step="10"
                         controls-position="right" />
      </FormField>
      <FormField label="查询改写" :hint="transformNote">
        <el-select v-model="settings.retrieval.query_transform">
          <el-option v-for="t in env.query_transforms" :key="t.type" :value="t.type" :label="t.label"
                     :disabled="!t.available">
            <span>{{ t.label }}</span><span v-if="!t.available" class="opt-note">需要大模型</span>
          </el-option>
        </el-select>
      </FormField>
    </div>

    <el-collapse v-if="part !== 'retrieval'" v-model="advancedOpen" class="advanced">
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
.enc-note {
  font-size: 12px;
  line-height: 1.45;
  color: var(--text-3);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 0 12px; }
.enrich { display: flex; flex-direction: column; gap: 8px; }
.switch-row { display: flex; align-items: flex-start; gap: 10px; font-size: 13px; line-height: 1.5; color: var(--text-2); cursor: pointer; }
.switch-row b { color: var(--text); font-weight: 500; }
.switch-row .el-switch { flex: none; margin-top: 1px; }
.switch-row .questions { flex: none; width: 76px; }
.enrich-cost { margin-top: 6px; font-size: 12px; color: var(--text-3); }
.weights { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.weight { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.weight-label { font-size: 12px; color: var(--text-2); }
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
