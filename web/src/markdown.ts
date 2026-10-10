// 分块正文按 Markdown 渲染：表格是真表格，有合并单元格的表格是 HTML，所以要开 html，再用 DOMPurify 消毒

import MarkdownIt from 'markdown-it'
import DOMPurify from 'dompurify'

const md = new MarkdownIt({ html: true, linkify: false, breaks: true })

export function renderMarkdown(text: string): string {
  return DOMPurify.sanitize(md.render(text))
}

const escape = (text: string) =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

/**
 * 渲染一个子块，开头 overlap 个字是和上一块重叠的句子，用 <mark class="overlap"> 标出来。
 * 重叠只发生在同一个长段落里，所以它就在第一个 <p> 的开头；对不上时退回单独渲染一段。
 */
export function renderChunk(body: string, overlap: number): string {
  if (!overlap) return renderMarkdown(body)
  const head = escape(body.slice(0, overlap))
  const html = renderMarkdown(body)
  if (html.startsWith(`<p>${head}`)) {
    return `<p><mark class="overlap">${head}</mark>${html.slice(3 + head.length)}`
  }
  return `<div class="overlap">${renderMarkdown(body.slice(0, overlap))}</div>${renderMarkdown(body.slice(overlap))}`
}
