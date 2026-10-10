<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ArrowLeft, Minus, Plus, WarningFilled } from '@element-plus/icons-vue'
import { api, type ChunkItem, type DocView, type PageLayout, type ParentItem } from '../api'
import { typeGroup } from '../format'
import { renderChunk } from '../markdown'
import ChunkInspector from '../components/ChunkInspector.vue'
import PageView from '../components/PageView.vue'

interface Heading { kind: 'heading'; key: string; text: string; depth: number; path: string[] }
interface Entry { kind: 'chunk'; key: string; chunk: ChunkItem; index: number; html: string }
interface Group { kind: 'group'; key: string; parentId: string | null; items: (Heading | Entry)[] }

const route = useRoute()
const router = useRouter()
const kbId = computed(() => String(route.params.id))
const docId = computed(() => String(route.params.docId))
const view = ref<DocView | null>(null)
const error = ref('')
const selectedId = ref('')
const returnTo = ref('')            // 从父块点进子块时记住父块，方便返回
const showParents = ref(true)
const showNumbers = ref(true)
const wide = ref(true)
const drawer = ref(false)
const activeHeading = ref('')
const body = ref<HTMLElement>()

// 原文视图：PDF、Word、PPT、Excel、图片按页面显示并框出分块；Markdown、CSV 等没有版面，只有文本视图
const mode = ref<'page' | 'text'>('page')
const pageLayout = ref<PageLayout | null>(null)
const layoutLoading = ref(false)
const showGaps = ref(true)
const zoom = ref(1)
const pageView = ref<InstanceType<typeof PageView>>()

async function load() {
  try {
    view.value = await api.document(kbId.value, docId.value)
    error.value = ''
  } catch (err) {
    error.value = (err as Error).message
    return
  }
  layoutLoading.value = true
  try {
    pageLayout.value = await api.layout(kbId.value, docId.value)
  } catch (err) {
    pageLayout.value = { available: false, reason: (err as Error).message, pages: [], regions: {}, methods: {},
                         uncovered: [], coverage: null }
  } finally {
    layoutLoading.value = false
  }
  if (!pageLayout.value.available) mode.value = 'text'
}

const pageReady = computed(() => mode.value === 'page' && !!pageLayout.value?.available)
const unlocated = computed(() => {
  const methods = pageLayout.value?.methods ?? {}
  return (view.value?.children ?? []).filter((c) => methods[c.id] === 'none')
})
const zoomBy = (step: number) => (zoom.value = Math.min(2, Math.max(0.6, Math.round((zoom.value + step) * 10) / 10)))
const METHOD_TEXT = { text: '文字层对齐', layout: '版面坐标（扫描页或公式）', none: '' } as const
const location = computed(() => {
  const item = selected.value
  const layout = pageLayout.value
  if (!item || item.parent || !layout?.available) return ''
  const method = layout.methods[item.item.id] ?? 'none'
  if (method === 'none') return '未能在原文中定位（如 PPT 备注、隐藏工作表、图片里的文字）'
  const pages = [...new Set((layout.regions[item.item.id] ?? []).map((r) => r.page))]
  return `第 ${pages.join('、')} 页 · ${METHOD_TEXT[method]}`
})
watch([kbId, docId], load, { immediate: true })

// ---------------------------------------------------------------- 把子块排成"标题 + 父块分组"
const parentNumber = computed(() => new Map((view.value?.parents ?? []).map((p, i) => [p.id, `P${i + 1}`])))
const childNumber = computed(() => new Map((view.value?.children ?? []).map((c, i) => [c.id, `#${i + 1}`])))

const layout = computed(() => {
  const blocks: (Heading | Group)[] = []
  const headings: Heading[] = []
  let previous: string[] = []
  let group: Group | null = null
  for (const [index, chunk] of (view.value?.children ?? []).entries()) {
    // 章节路径变化的部分渲染成标题
    const path = chunk.section ? chunk.section.split(' > ') : []
    let common = 0
    while (common < Math.min(path.length, previous.length) && path[common] === previous[common]) common++
    const fresh: Heading[] = path.slice(common).map((text, k) => ({
      kind: 'heading', key: `h${index}-${k}`, text, depth: common + k, path: path.slice(0, common + k + 1),
    }))
    previous = path
    headings.push(...fresh)
    // 同一个父块的相邻子块放进一组，左侧画一条竖线；新的一组开始前的标题放在组外
    const parentId = chunk.parent_id ?? null
    if (!group || parentId === null || group.parentId !== parentId) {
      blocks.push(...fresh)
      group = { kind: 'group', key: `g${index}`, parentId, items: [] }
      blocks.push(group)
    } else {
      group.items.push(...fresh)
    }
    group.items.push({ kind: 'chunk', key: chunk.id, chunk, index, html: renderChunk(chunk.body, chunk.overlap) })
  }
  return { blocks, headings }
})

// 目录：每个标题下有几个子块
const toc = computed(() => {
  const children = view.value?.children ?? []
  return layout.value.headings.map((h) => ({
    ...h,
    count: children.filter((c) => {
      const path = c.section ? c.section.split(' > ') : []
      return h.path.every((part, i) => path[i] === part)
    }).length,
  }))
})

// ---------------------------------------------------------------- 选中
const selected = computed<{ item: ChunkItem | ParentItem; parent: boolean } | null>(() => {
  const v = view.value
  if (!v || !selectedId.value) return null
  const child = v.children.find((c) => c.id === selectedId.value)
  if (child) return { item: child, parent: false }
  const parent = v.parents.find((p) => p.id === selectedId.value)
  return parent ? { item: parent, parent: true } : null
})

function select(id: string, from?: string) {
  returnTo.value = from ?? ''
  selectedId.value = id
  if (!wide.value) drawer.value = true
}

function selectFromInspector(id: string) {
  const fromParent = selected.value?.parent ? selected.value.item.id : undefined
  select(id, fromParent)
  scrollTo(id)
}

function scrollTo(id: string) {
  if (pageReady.value) {
    nextTick(() => pageView.value?.reveal(id))
    return
  }
  nextTick(() => {
    const isParent = view.value?.parents.some((p) => p.id === id)
    const el = document.querySelector<HTMLElement>(isParent ? `[data-parent="${CSS.escape(id)}"]` : `[data-chunk="${CSS.escape(id)}"]`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  })
}

function jump(key: string) {
  if (pageReady.value) {
    // 原文视图里跳到这一节第一个能定位的子块
    const heading = layout.value.headings.find((h) => h.key === key)
    const first = (view.value?.children ?? []).find((c) => {
      const path = c.section ? c.section.split(' > ') : []
      return heading?.path.every((part, i) => path[i] === part) && pageLayout.value?.methods[c.id] !== 'none'
    })
    if (first) {
      activeHeading.value = key
      pageView.value?.reveal(first.id)
    }
    return
  }
  document.getElementById(key)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

const budget = computed(() => {
  const c = view.value?.settings.chunker
  if (!c) return { size: 0, unit: 'token' }
  if (selected.value?.parent) return { size: c.parent_size || c.chunk_size, unit: 'token' }
  return { size: c.chunk_size, unit: c.type === 'structure' ? 'token' : '字符' }
})

// ---------------------------------------------------------------- 键盘 ↑ ↓ 切换子块
function onKey(event: KeyboardEvent) {
  const target = event.target as HTMLElement
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || !view.value?.children.length) return
  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
  event.preventDefault()
  const children = view.value.children
  const current = children.findIndex((c) => c.id === selectedId.value)
  const next = event.key === 'ArrowDown' ? Math.min(current + 1, children.length - 1) : Math.max(current - 1, 0)
  select(children[current < 0 ? 0 : next].id)
  scrollTo(children[current < 0 ? 0 : next].id)
}

// ---------------------------------------------------------------- 窄屏、目录高亮
const media = window.matchMedia('(min-width: 1280px)')
const onMedia = () => (wide.value = media.matches)
let observer: IntersectionObserver | null = null

watch(layout, () => {
  observer?.disconnect()
  nextTick(() => {
    observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible.length) activeHeading.value = visible[0].target.id
      },
      { rootMargin: '-60px 0px -70% 0px' },
    )
    body.value?.querySelectorAll('.heading').forEach((el) => observer?.observe(el))
  })
})

onMounted(() => {
  onMedia()
  media.addEventListener('change', onMedia)
  window.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => {
  media.removeEventListener('change', onMedia)
  window.removeEventListener('keydown', onKey)
  observer?.disconnect()
})
</script>

<template>
  <div class="doc">
    <header class="toolbar">
      <el-button text :icon="ArrowLeft" @click="router.push(`/kb/${kbId}`)">{{ view?.kb_name ?? '返回' }}</el-button>
      <span class="sep">/</span>
      <span v-if="view" class="type-badge" :class="`type-${typeGroup(view.file_type)}`">{{ view.file_type }}</span>
      <span class="doc-name" :title="docId">{{ docId }}</span>
      <div v-if="view" class="stats">
        <span>子块 <b>{{ view.stats.children }}</b></span>
        <span>父块 <b>{{ view.stats.parents }}</b></span>
        <span>平均 <b>{{ view.stats.avg_tokens }}</b> token</span>
        <span>最大 <b>{{ view.stats.max_tokens }}</b></span>
        <span v-if="view.parser" class="muted">{{ view.parser }}</span>
      </div>
      <div class="switches">
        <el-tooltip :disabled="!pageLayout || pageLayout.available" :content="pageLayout?.reason" placement="bottom">
          <el-segmented v-model="mode" size="small" :options="[
            { label: '原文', value: 'page', disabled: !!pageLayout && !pageLayout.available },
            { label: '文本', value: 'text' }]" />
        </el-tooltip>
        <el-switch v-model="showParents" size="small" active-text="父块边界" />
        <el-switch v-model="showNumbers" size="small" active-text="块编号" />
      </div>
    </header>

    <el-result v-if="error" icon="warning" title="打不开这篇文档" :sub-title="error" />

    <div v-else-if="view" class="columns" :class="{ wide }">
      <nav v-if="wide" class="toc">
        <div class="toc-title">目录</div>
        <a v-for="h in toc" :key="h.key" class="toc-item" :class="{ active: activeHeading === h.key }"
           :style="{ paddingLeft: `${10 + h.depth * 12}px` }" @click="jump(h.key)">
          <span class="toc-text">{{ h.text }}</span><span class="toc-count">{{ h.count }}</span>
        </a>
        <p v-if="!toc.length" class="muted toc-empty">这篇文档没有标题</p>
      </nav>

      <div v-if="mode === 'page' && (layoutLoading || pageReady)" class="body page-body">
        <div v-if="pageReady && pageLayout" class="page-bar">
          <span v-if="pageLayout.coverage !== null" class="cov" :class="{ low: pageLayout.coverage < 0.9 }">
            原文覆盖 <b>{{ (pageLayout.coverage * 100).toFixed(1) }}%</b>
          </span>
          <span v-else class="muted">没有文字层（扫描件），按解析器给出的版面坐标定位</span>
          <el-tooltip v-if="unlocated.length" placement="bottom"
                      :content="`未能在原文定位：${unlocated.map((c) => childNumber.get(c.id)).join(' ')}，可在文本视图查看`">
            <span class="warn"><el-icon><WarningFilled /></el-icon> {{ unlocated.length }} 块未定位</span>
          </el-tooltip>
          <el-switch v-model="showGaps" size="small" active-text="标出未进入分块的文字" />
          <span class="legend-gap" />
          <div class="zoom">
            <el-button text circle size="small" :icon="Minus" @click="zoomBy(-0.1)" />
            <span>{{ Math.round(zoom * 100) }}%</span>
            <el-button text circle size="small" :icon="Plus" @click="zoomBy(0.1)" />
          </div>
        </div>
        <PageView v-if="pageReady && pageLayout" ref="pageView" :kb-id="kbId" :doc-id="docId" :layout="pageLayout"
                  :children="view.children" :parents="view.parents" :selected-id="selectedId"
                  :show-parents="showParents" :show-numbers="showNumbers" :show-gaps="showGaps" :zoom="zoom"
                  @select="select" />
        <div v-else class="page-loading">
          <el-skeleton :rows="10" animated />
          <p class="hint">正在生成原文预览：渲染页面、在原文上定位每个分块。Word、PPT、Excel 第一次打开要先转换成 PDF，稍等几秒</p>
        </div>
      </div>

      <article v-else ref="body" class="body" :class="{ 'hide-parents': !showParents, 'hide-numbers': !showNumbers }">
        <template v-for="block in layout.blocks" :key="block.key">
          <template v-if="block.kind === 'heading'">
            <component :is="`h${Math.min(block.depth + 2, 5)}`" :id="block.key" class="heading"
                       :class="`d${Math.min(block.depth, 3)}`">{{ block.text }}</component>
          </template>
          <div v-else class="group" :class="{ parent: block.parentId }" :data-parent="block.parentId ?? undefined">
            <button v-if="block.parentId" type="button" class="bar"
                    :class="{ active: selectedId === block.parentId }"
                    :title="`父块 ${parentNumber.get(block.parentId)}：点击查看完整父块`"
                    @click="select(block.parentId!)">
              <span class="bar-label">{{ parentNumber.get(block.parentId) }}</span>
            </button>
            <template v-for="item in block.items" :key="item.key">
              <template v-if="item.kind === 'heading'">
                <component :is="`h${Math.min(item.depth + 2, 5)}`" :id="item.key" class="heading"
                           :class="`d${Math.min(item.depth, 3)}`">{{ item.text }}</component>
              </template>
              <div v-else class="chunk rise" :class="[`c${item.index % 6}`, { active: selectedId === item.chunk.id }]"
                   :style="{ '--i': Math.min(item.index, 24) }" :data-chunk="item.chunk.id"
                   @click="select(item.chunk.id)">
                <span class="no">#{{ item.index + 1 }}<span class="tk">{{ item.chunk.tokens }}</span></span>
                <div class="md" v-html="item.html" />
                <div v-if="item.chunk.assets.length" class="thumbs">
                  <img v-for="a in item.chunk.assets" :key="a.asset" :src="api.assetUrl(kbId, a.asset)"
                       :alt="a.caption || a.kind" loading="lazy" />
                </div>
              </div>
            </template>
          </div>
        </template>
      </article>

      <aside v-if="wide" class="side">
        <ChunkInspector :kb-id="kbId" :chunk="selected?.item ?? null" :is-parent="!!selected?.parent"
                        :number="selected?.parent ? parentNumber.get(selectedId) ?? '' : childNumber.get(selectedId) ?? ''"
                        :budget="budget.size" :unit="budget.unit"
                        :parent-label="parentNumber.get((selected?.item as ChunkItem | undefined)?.parent_id ?? '')"
                        :child-labels="Object.fromEntries(childNumber)"
                        :previous="returnTo ? `返回父块 ${parentNumber.get(returnTo)}` : ''" :location="location"
                        @select="selectFromInspector" @back="select(returnTo); scrollTo(selectedId)" />
      </aside>
    </div>
    <div v-else class="loading"><el-skeleton :rows="12" animated /></div>

    <el-drawer v-if="!wide" v-model="drawer" size="min(400px, 90vw)" :with-header="false">
      <ChunkInspector :kb-id="kbId" :chunk="selected?.item ?? null" :is-parent="!!selected?.parent"
                      :number="selected?.parent ? parentNumber.get(selectedId) ?? '' : childNumber.get(selectedId) ?? ''"
                      :budget="budget.size" :unit="budget.unit"
                      :parent-label="parentNumber.get((selected?.item as ChunkItem | undefined)?.parent_id ?? '')"
                      :child-labels="Object.fromEntries(childNumber)"
                      :previous="returnTo ? `返回父块 ${parentNumber.get(returnTo)}` : ''" :location="location"
                      @select="selectFromInspector" @back="select(returnTo); scrollTo(selectedId)" />
    </el-drawer>
  </div>
</template>

<style scoped>
.doc { min-height: 100%; }
.toolbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 20px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--bg) 92%, transparent);
  backdrop-filter: blur(10px);
}
.sep { color: var(--text-3); }
.doc-name { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 60px; }
.stats { display: flex; gap: 14px; margin-left: 12px; font-size: 12.5px; color: var(--text-2); white-space: nowrap; }
.stats b { font-weight: 600; color: var(--text); font-variant-numeric: tabular-nums; }
.switches { display: flex; gap: 14px; margin-left: auto; flex: none; }

.columns {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 24px;
  padding: 20px;
}
.columns.wide { grid-template-columns: 220px minmax(0, 1fr) 360px; }

.toc, .side {
  position: sticky;
  top: 72px;
  align-self: start;
  max-height: calc(100vh - 92px);
  overflow-y: auto;
}
.toc-title { font-size: 12px; font-weight: 600; color: var(--text-3); padding: 0 10px 8px; }
.toc-item {
  display: flex;
  gap: 8px;
  padding: 5px 10px;
  border-radius: 6px;
  color: var(--text-2);
  font-size: 13px;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}
.toc-item:hover { background: var(--surface-2); color: var(--text); }
.toc-item.active { color: var(--primary); background: color-mix(in srgb, var(--primary) 8%, transparent); }
.toc-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.toc-count { font-size: 11.5px; color: var(--text-3); font-variant-numeric: tabular-nums; }
.toc-empty { padding: 0 10px; font-size: 13px; }

.side { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }

.body { max-width: 860px; width: 100%; justify-self: center; padding-bottom: 40vh; }
.heading { margin: 22px 0 10px; scroll-margin-top: 80px; color: var(--text); }
.heading.d0 { font-size: 20px; }
.heading.d1 { font-size: 17px; }
.heading.d2 { font-size: 15px; }
.heading.d3 { font-size: 14px; color: var(--text-2); }
.body > .heading:first-child { margin-top: 4px; }

.group { position: relative; display: flex; flex-direction: column; gap: 6px; margin-bottom: 6px; }
.group.parent { padding-left: 18px; margin: 4px 0 10px; }
.bar {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 12px;
  padding: 0;
  border: none;
  background: none;
  cursor: pointer;
}
.bar::before {
  content: '';
  position: absolute;
  left: 4px;
  top: 2px;
  bottom: 2px;
  width: 3px;
  border-radius: 2px;
  background: var(--parent-bar);
  transition: background 0.2s ease, width 0.2s var(--ease);
}
.bar:focus { outline: none; }
.bar:hover::before, .bar.active::before, .bar:focus-visible::before { background: var(--primary); width: 4px; }
.bar-label {
  position: absolute;
  left: 12px;
  top: -2px;
  padding: 0 6px;
  border-radius: 4px;
  font: 600 11px/18px var(--mono);
  color: #fff;
  background: var(--primary);
  white-space: nowrap;
  opacity: 0;
  transform: translateX(-4px);
  transition: opacity 0.18s ease, transform 0.18s var(--ease);
  pointer-events: none;
  z-index: 2;
}
.bar:hover .bar-label, .bar.active .bar-label { opacity: 1; transform: none; }
.hide-parents .bar { display: none; }
.hide-parents .group.parent { padding-left: 0; }

.chunk {
  position: relative;
  padding: 12px 14px;
  border: 1px solid transparent;
  border-radius: 8px;
  color: var(--text);
  line-height: 1.75;
  cursor: pointer;
  transition: border-color 0.18s ease, box-shadow 0.2s var(--ease), transform 0.2s var(--ease);
}
.chunk:hover { border-color: var(--line); transform: translateY(-1px); box-shadow: var(--shadow); }
.chunk.active {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 16%, transparent);
}
.c0 { background: var(--c0); --line: var(--c0-line); }
.c1 { background: var(--c1); --line: var(--c1-line); }
.c2 { background: var(--c2); --line: var(--c2-line); }
.c3 { background: var(--c3); --line: var(--c3-line); }
.c4 { background: var(--c4); --line: var(--c4-line); }
.c5 { background: var(--c5); --line: var(--c5-line); }
.no {
  float: right;
  margin: -4px -6px 2px 10px;
  padding: 0 6px;
  border-radius: 4px;
  font: 11px/18px var(--mono);
  color: var(--text-3);
  background: color-mix(in srgb, var(--surface) 65%, transparent);
}
.tk { margin-left: 6px; opacity: 0.8; }
.tk::after { content: 't'; }
.hide-numbers .no { display: none; }
.chunk :deep(.overlap) {
  /* 斜纹要淡，文字始终清楚；下划虚线让重叠范围的起止一眼可见 */
  color: inherit;
  background: repeating-linear-gradient(-45deg, transparent 0 5px, color-mix(in srgb, var(--line) 70%, transparent) 5px 6.5px);
  border-bottom: 1px dashed color-mix(in srgb, var(--text-3) 60%, transparent);
  border-radius: 2px;
}
.thumbs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.thumbs img { max-height: 120px; max-width: 220px; border-radius: 6px; border: 1px solid var(--line); background: var(--surface); }
.loading { padding: 24px; max-width: 860px; margin: 0 auto; }
.page-body { max-width: none; }
.page-bar {
  position: sticky;
  top: 62px;
  z-index: 5;
  display: flex;
  align-items: center;
  gap: 16px;
  margin: 0 auto 14px;
  padding: 6px 14px;
  max-width: 860px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: color-mix(in srgb, var(--surface) 92%, transparent);
  backdrop-filter: blur(8px);
  font-size: 12.5px;
  color: var(--text-2);
}
.cov b { color: var(--success); font-variant-numeric: tabular-nums; }
.cov.low b { color: var(--warning); }
.warn { display: inline-flex; align-items: center; gap: 4px; color: var(--warning); cursor: default; }
.legend-gap {
  width: 22px;
  height: 12px;
  margin-left: -10px;
  border: 1px dashed rgba(212, 72, 59, 0.85);
  background: repeating-linear-gradient(-45deg, rgba(212, 72, 59, 0.2) 0 3px, transparent 3px 6px);
}
.zoom { display: flex; align-items: center; gap: 2px; margin-left: auto; font-variant-numeric: tabular-nums; }
.page-loading { max-width: 860px; margin: 0 auto; }
.page-loading .hint { margin-top: 12px; }
@media (max-width: 1080px) {
  .stats { display: none; }
}
@media (max-width: 700px) {
  .switches { display: none; }
}
</style>
