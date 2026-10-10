<script setup lang="ts">
import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { CopyDocument, Back } from '@element-plus/icons-vue'
import { api, type ChunkItem, type ParentItem } from '../api'
import { KIND_LABELS } from '../format'
import { renderMarkdown } from '../markdown'

const props = defineProps<{
  kbId: string
  chunk: ChunkItem | ParentItem | null
  number: string            // 页面上的编号，如 #7 或 P2
  isParent: boolean
  budget: number            // 子块（或父块）的大小预算
  unit: string
  parentLabel?: string      // 子块所属父块的编号
  childLabels?: Record<string, string>
  previous?: string         // 从父块点进子块时，返回父块的编号
}>()
const emit = defineEmits<{ (e: 'select', id: string): void; (e: 'back'): void }>()

const showMeta = ref(false)
const pages = computed(() => {
  const c = props.chunk
  if (!c?.page_start) return '—'
  return c.page_start === c.page_end ? `第 ${c.page_start} 页` : `第 ${c.page_start}–${c.page_end} 页`
})
const ratio = computed(() => (props.chunk ? Math.round((props.chunk.tokens / props.budget) * 100) : 0))
const over = computed(() => !!props.chunk && props.chunk.tokens > props.budget)
const json = computed(() => JSON.stringify(props.chunk?.metadata ?? {}, null, 2))
const parentId = computed(() => (props.chunk && !props.isParent ? (props.chunk as ChunkItem).parent_id : null))
const overlap = computed(() => (props.chunk && !props.isParent ? (props.chunk as ChunkItem).overlap : 0))
const children = computed(() => (props.isParent ? (props.chunk as ParentItem).children : []))

async function copy(text: string) {
  try {
    await navigator.clipboard.writeText(text)
    ElMessage.success('已复制')
  } catch {
    ElMessage.warning('浏览器不允许访问剪贴板')
  }
}
</script>

<template>
  <div class="inspector">
    <Transition name="fade" mode="out-in">
      <div v-if="chunk" :key="chunk.id" class="content">
        <div class="head">
          <el-button v-if="previous" text size="small" :icon="Back" @click="emit('back')">{{ previous }}</el-button>
          <h3>{{ isParent ? '父块' : '子块' }} <span class="num">{{ number }}</span></h3>
        </div>

        <div class="budget">
          <div class="budget-text">
            <b>{{ chunk.tokens }}</b> / {{ budget }} {{ unit }}
            <span class="muted">· {{ chunk.chars }} 字符</span>
          </div>
          <el-progress :percentage="Math.min(ratio, 100)" :stroke-width="6" :show-text="false"
                       :status="over ? 'warning' : undefined" />
          <div v-if="over" class="hint">超出预算：公式、图片不切分，表格的一行、代码的一行也不会从中间断开</div>
        </div>

        <dl class="props">
          <dt>ID</dt>
          <dd class="id">
            <span class="mono">{{ chunk.id }}</span>
            <el-button text circle size="small" :icon="CopyDocument" @click="copy(chunk.id)" />
          </dd>
          <template v-if="!isParent">
            <dt>父块</dt>
            <dd>
              <el-link v-if="parentId" type="primary" @click="emit('select', parentId)">{{ parentLabel }}（点击查看）</el-link>
              <span v-else class="muted">无（文档或小节较短，不生成父块）</span>
            </dd>
          </template>
          <dt>章节</dt>
          <dd>{{ chunk.section || '—' }}</dd>
          <dt>页码</dt>
          <dd>{{ pages }}</dd>
          <dt>元素</dt>
          <dd class="kinds">
            <el-tag v-for="k in chunk.kinds" :key="k" size="small" effect="plain">{{ KIND_LABELS[k] ?? k }}</el-tag>
            <span v-if="!chunk.kinds.length" class="muted">—</span>
          </dd>
          <template v-if="overlap">
            <dt>重叠</dt>
            <dd>开头 {{ overlap }} 个字符与上一块重复</dd>
          </template>
        </dl>

        <div v-if="isParent" class="block">
          <div class="block-title">包含的子块（{{ children.length }}）</div>
          <div class="chips">
            <button v-for="id in children" :key="id" class="chip" @click="emit('select', id)">
              {{ childLabels?.[id] ?? id }}
            </button>
          </div>
          <div class="block-title">父块全文（交给大模型的上下文）</div>
          <div class="parent-text md" v-html="renderMarkdown(chunk.body)" />
        </div>

        <div v-if="chunk.assets.length" class="block">
          <div class="block-title">图片资产（{{ chunk.assets.length }}）</div>
          <div v-for="a in chunk.assets" :key="a.asset" class="asset">
            <el-image :src="api.assetUrl(kbId, a.asset)" :preview-src-list="[api.assetUrl(kbId, a.asset)]"
                      fit="contain" class="asset-img" preview-teleported>
              <template #error><div class="asset-missing">资产库中找不到原图</div></template>
            </el-image>
            <div class="asset-meta">
              <div>{{ a.caption || '（无图注）' }}</div>
              <div class="muted">
                {{ KIND_LABELS[a.kind] ?? a.kind }}<template v-if="a.page"> · 第 {{ a.page }} 页</template>
                <template v-if="a.bbox"> · bbox [{{ a.bbox.map((v) => v.toFixed(3)).join(', ') }}]</template>
              </div>
            </div>
          </div>
        </div>

        <div class="block">
          <div class="block-title meta-title">
            <el-button link size="small" @click="showMeta = !showMeta">{{ showMeta ? '收起' : '展开' }}原始元数据</el-button>
            <el-button v-if="showMeta" link size="small" :icon="CopyDocument" @click="copy(json)">复制</el-button>
          </div>
          <el-collapse-transition>
            <pre v-show="showMeta" class="json">{{ json }}</pre>
          </el-collapse-transition>
        </div>
      </div>

      <div v-else class="empty">
        <h3>点击任意分块查看详情</h3>
        <ul class="legend">
          <li><span class="sw sw-colors"><i /><i /><i /></span>相邻子块用不同底色区分</li>
          <li><span class="sw sw-bar" />左侧竖线框出同一个父块</li>
          <li><span class="sw sw-hatch" />斜纹是与上一块重叠的句子</li>
          <li><kbd>↑</kbd><kbd>↓</kbd>在分块之间切换</li>
        </ul>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.inspector { padding: 18px; }
.head { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; margin-bottom: 14px; }
.head h3 { font-size: 16px; }
.num { color: var(--primary); font-variant-numeric: tabular-nums; }
.budget { margin-bottom: 16px; }
.budget-text { font-size: 13px; margin-bottom: 6px; }
.budget-text b { font-size: 18px; font-weight: 600; }
.props {
  display: grid;
  grid-template-columns: 44px minmax(0, 1fr);
  gap: 8px 12px;
  margin: 0 0 16px;
  font-size: 13px;
}
.props dt { color: var(--text-3); }
.props dd { margin: 0; word-break: break-all; }
.id { display: flex; align-items: flex-start; gap: 4px; }
.id .mono { font-size: 12px; color: var(--text-2); }
.kinds { display: flex; flex-wrap: wrap; gap: 4px; }
.block { border-top: 1px solid var(--border); padding-top: 12px; margin-top: 12px; }
.block-title { font-size: 12px; font-weight: 600; color: var(--text-3); margin-bottom: 8px; }
.meta-title { display: flex; justify-content: space-between; margin-bottom: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.chip {
  padding: 2px 10px;
  border: 1px solid var(--border-strong);
  border-radius: 12px;
  background: var(--surface);
  color: var(--text-2);
  font: 12px var(--mono);
  cursor: pointer;
  transition: border-color 0.15s ease, color 0.15s ease;
}
.chip:hover { border-color: var(--primary); color: var(--primary); }
.parent-text {
  max-height: 340px;
  overflow-y: auto;
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--surface-2);
  font-size: 13px;
}
.asset { display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; }
.asset-img { width: 100%; max-height: 200px; border-radius: 6px; background: var(--surface-2); }
.asset-missing { padding: 24px; text-align: center; font-size: 12px; color: var(--text-3); }
.asset-meta { font-size: 12.5px; }
.json {
  max-height: 360px;
  overflow: auto;
  margin: 8px 0 0;
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--surface-2);
  font-size: 12px;
  line-height: 1.5;
}
.empty { padding: 24px 6px; color: var(--text-2); }
.empty h3 { font-size: 15px; margin-bottom: 14px; color: var(--text); }
.legend { list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; font-size: 13px; }
.legend li { display: flex; align-items: center; gap: 10px; }
.sw { flex: none; width: 34px; height: 18px; border-radius: 4px; }
.sw-colors { display: flex; gap: 2px; }
.sw-colors i { flex: 1; border-radius: 3px; }
.sw-colors i:nth-child(1) { background: var(--c0); }
.sw-colors i:nth-child(2) { background: var(--c1); }
.sw-colors i:nth-child(3) { background: var(--c2); }
.sw-bar { width: 34px; border-left: 3px solid var(--parent-bar); border-radius: 0; height: 22px; }
.sw-hatch { background: var(--c0) repeating-linear-gradient(-45deg, transparent 0 5px, var(--c0-line) 5px 6.5px); border-bottom: 1px dashed var(--text-3); border-radius: 2px; }
kbd {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  height: 20px;
  padding: 0 5px;
  border: 1px solid var(--border-strong);
  border-bottom-width: 2px;
  border-radius: 4px;
  font: 11px var(--mono);
  color: var(--text-2);
}
.legend li kbd + kbd { margin-left: -6px; }
</style>
