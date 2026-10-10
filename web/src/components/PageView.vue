<script setup lang="ts">
import { computed, ref } from 'vue'
import { api, type ChunkItem, type PageLayout, type ParentItem } from '../api'

// 原文页面打底，在上面框出每个分块：子块用和文本视图相同的 6 种颜色轮换，父块用虚线框出范围，
// 没有进入任何分块的文字用红色斜纹标出，一眼能看出分块是否完整、切口落在原文的哪里。
const props = defineProps<{
  kbId: string
  docId: string
  layout: PageLayout
  children: ChunkItem[]
  parents: ParentItem[]
  selectedId: string
  showParents: boolean
  showNumbers: boolean
  showGaps: boolean
  zoom: number
}>()
const emit = defineEmits<{ (e: 'select', id: string): void }>()

const hovered = ref('')
const root = ref<HTMLElement>()

interface Box { key: string; id: string; index: number; bbox: number[]; first: boolean }
interface Frame { key: string; id: string; label: string; bbox: number[] }

const index = computed(() => new Map(props.children.map((c, i) => [c.id, i])))
const parentOf = computed(() => new Map(props.children.map((c) => [c.id, c.parent_id ?? ''])))

// 按页分好：子块的框、父块的外框（它的子块在这一页上的框的外接矩形）、未覆盖的文字
const pages = computed(() => {
  const out = props.layout.pages.map(() => ({ boxes: [] as Box[], frames: [] as Frame[], gaps: [] as number[][] }))
  const parentBoxes = new Map<string, Map<number, number[]>>()
  for (const child of props.children) {
    const regions = props.layout.regions[child.id] ?? []
    const seen = new Set<number>()
    regions.forEach((region, k) => {
      const page = out[region.page - 1]
      if (!page) return
      page.boxes.push({ key: `${child.id}-${k}`, id: child.id, index: index.value.get(child.id) ?? 0,
                        bbox: region.bbox, first: !seen.has(region.page) })
      seen.add(region.page)
      const parent = parentOf.value.get(child.id)
      if (parent) {
        const byPage = parentBoxes.get(parent) ?? new Map<number, number[]>()
        const [x0, y0, x1, y1] = region.bbox
        const box = byPage.get(region.page)
        byPage.set(region.page, box ? [Math.min(box[0], x0), Math.min(box[1], y0), Math.max(box[2], x1), Math.max(box[3], y1)]
          : [x0, y0, x1, y1])
        parentBoxes.set(parent, byPage)
      }
    })
  }
  props.parents.forEach((parent, i) => {
    for (const [page, bbox] of parentBoxes.get(parent.id) ?? []) {
      out[page - 1]?.frames.push({ key: `${parent.id}-${page}`, id: parent.id, label: `P${i + 1}`, bbox })
    }
  })
  for (const gap of props.layout.uncovered) out[gap.page - 1]?.gaps.push(gap.bbox)
  return out
})

// 文字行的框上下各留一点空，看起来像荧光笔划过，而不是贴着字的细条
function place(bbox: number[], pad = 0.002) {
  const [x0, y0, x1, y1] = bbox
  return {
    left: `${(x0 - pad) * 100}%`,
    top: `${(y0 - pad) * 100}%`,
    width: `${(x1 - x0 + pad * 2) * 100}%`,
    height: `${(y1 - y0 + pad * 2) * 100}%`,
  }
}

// 页面按实际大小等比例显示：Excel 每个工作表导出成一页，内容少的表页面很小，不能和大页面一样撑满宽度
const maxWidth = computed(() => Math.max(1, ...props.layout.pages.map((p) => p.width)))
const pageUrl = (n: number) => `${api.pageUrl(props.kbId, props.docId, n)}&v=${props.layout.version ?? ''}`
const active = (id: string) => id === props.selectedId || props.selectedId === parentOf.value.get(id)

/** 滚动到分块在原文上的第一个框；没能定位的块返回 false */
function reveal(id: string): boolean {
  const el = root.value?.querySelector<HTMLElement>(`[data-box="${CSS.escape(id)}"], [data-frame="${CSS.escape(id)}"]`)
  el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  return !!el
}
defineExpose({ reveal })
</script>

<template>
  <div ref="root" class="pages" :style="{ maxWidth: `${Math.round(860 * zoom)}px` }">
    <section v-for="(page, i) in layout.pages" :key="i" class="page rise" :style="{ '--i': Math.min(i, 6),
             aspectRatio: `${page.width} / ${page.height}`, width: `${(page.width / maxWidth) * 100}%` }"
             :data-page="i + 1">
      <img :src="pageUrl(i + 1)" :alt="`第 ${i + 1} 页`" loading="lazy" draggable="false" />
      <div class="overlay" @mouseleave="hovered = ''">
        <template v-if="showGaps">
          <div v-for="(gap, k) in pages[i].gaps" :key="`gap-${k}`" class="gap" :style="place(gap)"
               title="没有进入任何分块的文字（页眉页脚是解析时故意丢弃的，其余多半是解析遗漏）" />
        </template>
        <template v-if="showParents">
          <div v-for="frame in pages[i].frames" :key="frame.key" class="frame" :data-frame="frame.id"
               :class="{ active: selectedId === frame.id }" :style="place(frame.bbox, 0.008)"
               @click.self="emit('select', frame.id)">
            <span class="frame-label" @click="emit('select', frame.id)">{{ frame.label }}</span>
          </div>
        </template>
        <div v-for="box in pages[i].boxes" :key="box.key" class="box" :data-box="box.first ? box.id : undefined"
             :class="[`c${box.index % 6}`, { active: selectedId === box.id, hover: hovered === box.id,
                       dim: selectedId && !active(box.id) }]"
             :style="place(box.bbox)" @mouseenter="hovered = box.id" @click="emit('select', box.id)" />
        <!-- 编号单独一层：放在框里会跟着框做 multiply 混合，白字就看不见了 -->
        <template v-if="showNumbers">
          <span v-for="box in pages[i].boxes.filter((b) => b.first)" :key="`label-${box.key}`" class="label"
                :class="[`c${box.index % 6}`, { dim: selectedId && !active(box.id) }]"
                :style="{ left: place(box.bbox).left, top: place(box.bbox).top }">#{{ box.index + 1 }}</span>
        </template>
      </div>
      <div class="page-no">{{ i + 1 }} / {{ layout.pages.length }}</div>
    </section>
  </div>
</template>

<style scoped>
.pages { width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; padding-bottom: 30vh; }
.page {
  position: relative;
  align-self: center;
  background: #fff;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12), 0 8px 24px rgba(15, 23, 42, 0.08);
  overflow: hidden;
}
.page img { position: absolute; inset: 0; width: 100%; height: 100%; user-select: none; }
.overlay { position: absolute; inset: 0; }
.page-no {
  position: absolute;
  right: 8px;
  bottom: 6px;
  padding: 0 6px;
  border-radius: 4px;
  font: 11px/18px var(--mono);
  color: #4e5969;
  background: rgba(255, 255, 255, 0.85);
}

/* 子块：半透明荧光色 + multiply 混合，纸面上的黑字始终清楚 */
.box {
  position: absolute;
  border: 1px solid var(--o-line);
  border-radius: 2px;
  background: var(--o-fill);
  mix-blend-mode: multiply;
  cursor: pointer;
  transition: background 0.15s ease, opacity 0.2s ease, box-shadow 0.15s ease;
}
.box.hover { background: var(--o-hover); }
.box.active { background: var(--o-hover); box-shadow: 0 0 0 2px var(--o-line); }
.box.dim { opacity: 0.45; }
.c0 { --o-fill: rgba(59, 110, 246, 0.16); --o-hover: rgba(59, 110, 246, 0.3); --o-line: rgba(59, 110, 246, 0.75); --o-solid: #3b6ef6; }
.c1 { --o-fill: rgba(22, 163, 74, 0.16); --o-hover: rgba(22, 163, 74, 0.3); --o-line: rgba(22, 163, 74, 0.75); --o-solid: #16a34a; }
.c2 { --o-fill: rgba(234, 140, 22, 0.18); --o-hover: rgba(234, 140, 22, 0.32); --o-line: rgba(234, 140, 22, 0.8); --o-solid: #d97f0e; }
.c3 { --o-fill: rgba(139, 92, 246, 0.16); --o-hover: rgba(139, 92, 246, 0.3); --o-line: rgba(139, 92, 246, 0.75); --o-solid: #7c4ddf; }
.c4 { --o-fill: rgba(13, 148, 168, 0.16); --o-hover: rgba(13, 148, 168, 0.3); --o-line: rgba(13, 148, 168, 0.75); --o-solid: #0d8a9c; }
.c5 { --o-fill: rgba(225, 29, 92, 0.14); --o-hover: rgba(225, 29, 92, 0.28); --o-line: rgba(225, 29, 92, 0.7); --o-solid: #d61f5a; }
.label {
  position: absolute;
  transform: translateY(-100%);
  transition: opacity 0.2s ease;
  padding: 0 4px;
  border-radius: 3px 3px 3px 0;
  font: 600 10px/14px var(--mono);
  color: #fff;
  background: var(--o-solid);
  white-space: nowrap;
  pointer-events: none;
}
.label.dim { opacity: 0.4; }

/* 父块：虚线外框，标签在右上角 */
.frame {
  position: absolute;
  border: 1.5px dashed rgba(78, 89, 105, 0.55);
  border-radius: 4px;
  transition: border-color 0.15s ease;
}
.frame.active { border: 2px solid var(--primary); }
.frame-label {
  position: absolute;
  right: -1px;
  top: 0;
  transform: translateY(-100%);
  padding: 0 5px;
  border-radius: 3px 3px 0 3px;
  font: 600 10px/14px var(--mono);
  color: #fff;
  background: rgba(78, 89, 105, 0.8);
  cursor: pointer;
}
.frame.active .frame-label { background: var(--primary); }

/* 没有进入任何分块的文字 */
.gap {
  position: absolute;
  border: 1px dashed rgba(212, 72, 59, 0.85);
  background: repeating-linear-gradient(-45deg, rgba(212, 72, 59, 0.16) 0 4px, rgba(212, 72, 59, 0.04) 4px 8px);
  mix-blend-mode: multiply;
  pointer-events: auto;
}
</style>
