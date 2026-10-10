// 后端接口（learn_rag/server/app.py）的类型和调用。所有请求都走这里，页面不直接拼 URL。

export interface ChunkerSettings {
  type: 'structure' | 'recursive' | 'fixed'
  chunk_size: number
  parent_size?: number
  overlap_sentences?: number
  chunk_overlap?: number
}

export interface Settings {
  chunker: ChunkerSettings
  encoder: { type: string; model: string }
  index: { type: 'chroma' | 'flat'; space?: string; ef_construction?: number; max_neighbors?: number; ef_search?: number }
  parsers: Record<string, string[]>
  ocr: boolean
}

export interface EncoderOption {
  type: string
  label: string
  model: string
  available: boolean
  note: string
}

export interface ParserChoice {
  value: string[]
  label: string
  available: boolean
  note?: string
}

export interface EnvReport {
  encoders: EncoderOption[]
  keys: { name: string; label: string; configured: boolean }[]
  packages: { name: string; installed: boolean; install: string }[]
  parsers: Record<string, ParserChoice[]>
  ocr: boolean
  defaults: Settings
}

export interface StagedFile {
  path: string
  size: number
  type: string
  supported: boolean
}

export interface Recommendation {
  summary: string[]
  settings: Settings
  reasons: Record<string, string>
  warnings: string[]
}

export interface Verify {
  ok: boolean
  problems: string[]
  documents: number
  chunks: number
  bm25_chunks: number
  parents: number
  sampled: number
}

export type FileStatus = 'waiting' | 'parsing' | 'embedding' | 'done' | 'skipped' | 'failed'

export interface FileProgress {
  path: string
  status: FileStatus
  chunks: number
  parents: number
  message: string
}

export interface Job {
  id: string
  kb_id: string
  kind: 'build' | 'append' | 'rebuild'
  status: 'queued' | 'running' | 'done' | 'failed'
  error: string
  verify: Verify | null
  files: FileProgress[]
  counts: Record<FileStatus, number>
  processed: number
  created_at: number
  finished_at: number | null
  version: number
}

export interface KbSummary {
  id: string
  name: string
  settings: Settings
  created_at: number
  updated_at: number
  files: number
  chunks: number
  parents: number
  failed: number
  busy: boolean
}

export interface DocRow {
  doc_id: string
  type: string
  size: number
  status: FileStatus
  message: string
  chunks: number
  parents: number
  parser?: string
  pages?: number | null
  uploaded_at: number
  updated_at?: number
}

export interface KbInfo extends KbSummary {
  documents: DocRow[]
  verify: Verify | null
  job: Job | null
}

export interface Asset {
  asset: string
  kind: string
  caption?: string
  page?: number
  bbox?: number[]
  mime?: string
}

export interface ChunkItem {
  id: string
  position: number
  body: string
  section: string
  tokens: number
  chars: number
  kinds: string[]
  page_start?: number | null
  page_end?: number | null
  assets: Asset[]
  metadata: Record<string, unknown>
  parent_id?: string | null
  overlap: number
}

export interface ParentItem extends Omit<ChunkItem, 'parent_id' | 'overlap'> {
  children: string[]
}

export interface DocView {
  doc_id: string
  title: string
  file_type: string
  parser: string
  pages: number | null
  stats: { children: number; parents: number; tokens: number; avg_tokens: number; max_tokens: number }
  children: ChunkItem[]
  parents: ParentItem[]
  kb_id: string
  kb_name: string
  settings: Settings
  file: Partial<DocRow>
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method }
  if (body instanceof FormData) init.body = body
  else if (body !== undefined) {
    init.body = JSON.stringify(body)
    init.headers = { 'Content-Type': 'application/json' }
  }
  let response: Response
  try {
    response = await fetch(url, init)
  } catch {
    throw new Error('连不上后端服务，请确认已运行 learn-rag serve')
  }
  const text = await response.text()
  const data = text ? JSON.parse(text) : null
  if (!response.ok) {
    const detail = data?.detail
    throw new Error(typeof detail === 'string' ? detail : `请求失败（HTTP ${response.status}）`)
  }
  return data as T
}

// 文档 id 是带目录的相对路径，每一段分别编码，斜杠保留
const docPath = (docId: string) => docId.split('/').map(encodeURIComponent).join('/')

export const api = {
  env: () => request<EnvReport>('GET', '/api/env'),

  upload(files: { file: File; path: string }[], uploadId?: string) {
    const form = new FormData()
    for (const { file, path } of files) {
      form.append('files', file, file.name)
      form.append('paths', path)
    }
    if (uploadId) form.append('upload_id', uploadId)
    return request<{ upload_id: string; files: StagedFile[] }>('POST', '/api/uploads', form)
  },
  unstage: (uploadId: string, path: string) =>
    request<{ files: StagedFile[] }>('DELETE', `/api/uploads/${uploadId}/files?path=${encodeURIComponent(path)}`),
  recommend: (uploadId: string) => request<Recommendation>('POST', `/api/uploads/${uploadId}/recommend`),

  kbs: () => request<KbSummary[]>('GET', '/api/kbs'),
  createKb: (name: string, uploadId: string, settings: Settings) =>
    request<{ kb: KbInfo; job: Job }>('POST', '/api/kbs', { name, upload_id: uploadId, settings }),
  kb: (id: string) => request<KbInfo>('GET', `/api/kbs/${id}`),
  deleteKb: (id: string) => request<{ ok: boolean }>('DELETE', `/api/kbs/${id}`),
  addFiles: (id: string, uploadId: string) => request<Job>('POST', `/api/kbs/${id}/files`, { upload_id: uploadId }),
  rebuild: (id: string, settings?: Settings) => request<Job>('POST', `/api/kbs/${id}/rebuild`, { settings: settings ?? null }),

  document: (id: string, docId: string) => request<DocView>('GET', `/api/kbs/${id}/documents/${docPath(docId)}`),
  deleteDocument: (id: string, docId: string) =>
    request<{ ok: boolean }>('DELETE', `/api/kbs/${id}/documents/${docPath(docId)}`),
  assetUrl: (id: string, asset: string) => `/api/kbs/${id}/assets/${asset.split('/').map(encodeURIComponent).join('/')}`,
}

/**
 * 订阅任务进度：优先用 SSE，连接断开时退回每秒轮询，任务结束后自动停止。返回取消函数。
 */
export function watchJob(jobId: string, onUpdate: (job: Job) => void): () => void {
  let stopped = false
  let timer: number | undefined
  const source = new EventSource(`/api/jobs/${jobId}/events`)
  const poll = async () => {
    if (stopped) return
    try {
      const job = await request<Job>('GET', `/api/jobs/${jobId}`)
      onUpdate(job)
      if (job.status === 'done' || job.status === 'failed') return
    } catch {
      /* 后端暂时不可用，继续轮询 */
    }
    timer = window.setTimeout(poll, 1000)
  }
  source.onmessage = (event) => {
    const job = JSON.parse(event.data) as Job
    onUpdate(job)
    if (job.status === 'done' || job.status === 'failed') source.close()
  }
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED || stopped) return
    source.close()
    poll()
  }
  return () => {
    stopped = true
    source.close()
    window.clearTimeout(timer)
  }
}
