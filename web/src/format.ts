// 页面上反复用到的展示格式

import type { FileStatus, Settings } from './api'

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

export function timeAgo(seconds?: number | null): string {
  if (!seconds) return '—'
  const diff = Date.now() / 1000 - seconds
  if (diff < 60) return '刚刚'
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`
  const d = new Date(seconds * 1000)
  return `${d.getMonth() + 1}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

export const CHUNKER_LABELS: Record<string, string> = {
  structure: '结构感知父子分块',
  recursive: '递归字符切分',
  fixed: '定长滑窗',
}

export function chunkerSummary(s: Settings): string {
  const c = s.chunker
  if (c.type === 'structure') {
    const parts = [`子块 ${c.chunk_size}`, c.parent_size ? `父块 ${c.parent_size}` : '无父块']
    if (c.overlap_sentences) parts.push(`重叠 ${c.overlap_sentences} 句`)
    return parts.join(' · ')
  }
  return `${CHUNKER_LABELS[c.type]} · ${c.chunk_size} 字 · 重叠 ${c.chunk_overlap ?? 0}`
}

/** 开了哪些建库增强；全关时返回空串 */
export function enrichSummary(s: Settings): string {
  const e = s.enrich
  if (!e) return ''
  const parts = [e.context && '背景', e.describe && '图表描述', e.questions > 0 && `${e.questions} 个假设问题`,
                 e.summaries && '章节摘要'].filter(Boolean)
  return parts.length ? `增强：${parts.join(' · ')}` : ''
}

export function encoderSummary(s: Settings): string {
  return s.encoder.type === 'hashing' ? 'Hashing' : s.encoder.model
}

/** 能同时生成稀疏向量、可以开稀疏召回的 embedding */
export const SPARSE_ENCODERS = ['bge_m3']

const RERANKER_LABELS: Record<string, string> = { lexical: '词重叠精排', identity: '不精排' }
const CHANNEL_LABELS: Record<string, string> = { vector: '向量', bm25: 'BM25', sparse: '稀疏' }
const TRANSFORM_LABELS: Record<string, string> = { multi_query: '多查询改写', hyde: 'HyDE' }

export function rerankerSummary(s: Settings): string {
  const r = s.retrieval
  const channels = Object.entries(r.weights).filter(([, w]) => w > 0).map(([c]) => CHANNEL_LABELS[c] ?? c)
  const name = RERANKER_LABELS[s.reranker.type] ?? `精排 ${s.reranker.model.split('/').pop()}`
  const parts = [channels.join(' + '), name, `${r.candidate_k} → ${r.top_k}`]
  if (TRANSFORM_LABELS[r.query_transform]) parts.unshift(TRANSFORM_LABELS[r.query_transform])
  return parts.join(' · ')
}

export const STATUS: Record<FileStatus, { label: string; type: 'info' | 'primary' | 'success' | 'warning' | 'danger' }> = {
  waiting: { label: '等待中', type: 'info' },
  parsing: { label: '解析中', type: 'primary' },
  embedding: { label: '向量化中', type: 'primary' },
  done: { label: '完成', type: 'success' },
  skipped: { label: '未变化', type: 'info' },
  failed: { label: '失败', type: 'danger' },
}

export const KIND_LABELS: Record<string, string> = {
  text: '段落',
  table: '表格',
  image: '图片',
  formula: '公式',
  code: '代码',
  heading: '标题',
}

const TYPE_GROUPS: Record<string, string> = {
  pdf: 'pdf', doc: 'word', docx: 'word', rtf: 'word', xls: 'excel', xlsx: 'excel', csv: 'excel', tsv: 'excel',
  ppt: 'ppt', pptx: 'ppt', md: 'text', markdown: 'text', txt: 'text', html: 'web', htm: 'web',
  png: 'image', jpg: 'image', jpeg: 'image', webp: 'image',
}

/** 文件类型徽标的配色分组 */
export function typeGroup(type: string): string {
  return TYPE_GROUPS[type.toLowerCase()] ?? 'other'
}

/** 前端先拦一道明显不合法的参数（后端还会再校验一次），返回错误信息，没问题返回空字符串 */
/** 表单里第一处不合法的参数；空串表示可以提交 */
export function settingsError(s: Settings): string {
  return chunkerError(s) || retrievalError(s)
}

export function chunkerError(s: Settings): string {
  const c = s.chunker
  if (!c.chunk_size || c.chunk_size < 20) return '子块大小至少 20'
  if (c.type === 'structure' && c.parent_size && c.parent_size <= c.chunk_size) return '父块必须大于子块；设为 0 表示不生成父块'
  if (c.type !== 'structure' && (c.chunk_overlap ?? 0) >= c.chunk_size) return '重叠必须小于块大小'
  return ''
}

export function retrievalError(s: Settings): string {
  const w = s.retrieval.weights
  if (!(w.vector > 0 || w.bm25 > 0 || w.sparse > 0)) return '至少要有一路召回的权重大于 0'
  if (w.sparse > 0 && !SPARSE_ENCODERS.includes(s.encoder.type)) return '稀疏召回需要 embedding 选 BGE-M3（稠密 + 稀疏）'
  return ''
}

/** 深拷贝纯数据。不能用 structuredClone：响应式对象是 Proxy，structuredClone 会报错 */
export function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value))
}
