<script setup lang="ts">
import { onMounted } from 'vue'
import { Plus, Loading } from '@element-plus/icons-vue'
import { store, refreshKbs } from '../store'
import { chunkerSummary, encoderSummary, timeAgo } from '../format'

onMounted(refreshKbs)

const steps = [
  { title: '上传文件', text: '拖入文件或整个文件夹，支持 PDF、Word、PPT、Excel、Markdown、网页和图片。' },
  { title: '选择参数', text: '子块、父块、重叠、embedding 模型都在页面上设置，也可以一键推荐。' },
  { title: '建库并查看', text: '后台逐个文件解析、切分、向量化，完成后点开文档查看每一个分块。' },
]
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1>知识库</h1>
        <div class="sub">每个知识库有自己的切分参数和 embedding 模型，互不影响</div>
      </div>
      <RouterLink v-if="store.kbs.length" to="/new" custom v-slot="{ navigate }">
        <el-button type="primary" :icon="Plus" @click="navigate">新建知识库</el-button>
      </RouterLink>
    </div>

    <div v-if="store.kbs.length" class="grid">
      <RouterLink v-for="(kb, i) in store.kbs" :key="kb.id" :to="`/kb/${kb.id}`" class="card kb-card rise"
                  :style="{ '--i': i }">
        <div class="kb-top">
          <h3>{{ kb.name }}</h3>
          <el-tag v-if="kb.busy" size="small" round>
            <el-icon class="spin"><Loading /></el-icon> 建库中
          </el-tag>
          <el-tag v-else-if="kb.failed" size="small" type="danger" round>{{ kb.failed }} 个失败</el-tag>
        </div>
        <div class="kb-params">
          <span>{{ chunkerSummary(kb.settings) }}</span>
          <span>{{ encoderSummary(kb.settings) }} · {{ kb.settings.index.type === 'chroma' ? 'Chroma' : 'Flat' }}</span>
        </div>
        <div class="kb-stats">
          <div><b>{{ kb.files }}</b><span>文档</span></div>
          <div><b>{{ kb.chunks }}</b><span>子块</span></div>
          <div><b>{{ kb.parents }}</b><span>父块</span></div>
        </div>
        <div class="kb-foot">更新于 {{ timeAgo(kb.updated_at) }}</div>
      </RouterLink>
    </div>

    <div v-else-if="store.kbsLoaded" class="card welcome rise">
      <h2>从上传文件开始</h2>
      <p class="muted">不需要记命令，也不需要改配置文件。三步建好一个知识库：</p>
      <ol class="steps">
        <li v-for="(step, i) in steps" :key="step.title" class="rise" :style="{ '--i': i + 1 }">
          <span class="num">{{ i + 1 }}</span>
          <div><b>{{ step.title }}</b><p>{{ step.text }}</p></div>
        </li>
      </ol>
      <RouterLink to="/new" custom v-slot="{ navigate }">
        <el-button type="primary" size="large" :icon="Plus" @click="navigate">新建知识库</el-button>
      </RouterLink>
      <p v-if="!store.online" class="offline">后端未连接：请先在项目根目录运行 <code>learn-rag serve</code></p>
    </div>
  </div>
</template>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}
.kb-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 18px;
  color: var(--text);
  transition: transform 0.22s var(--ease), box-shadow 0.22s var(--ease), border-color 0.22s ease;
}
.kb-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-hover);
  border-color: color-mix(in srgb, var(--primary) 35%, var(--border));
}
.kb-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.kb-top h3 { font-size: 16px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kb-params { display: flex; flex-direction: column; gap: 2px; font-size: 12.5px; color: var(--text-2); }
.kb-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  padding: 10px 0;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.kb-stats div { display: flex; flex-direction: column; }
.kb-stats b { font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums; }
.kb-stats span { font-size: 12px; color: var(--text-3); }
.kb-foot { font-size: 12px; color: var(--text-3); }
.spin { animation: spin 1s linear infinite; vertical-align: -2px; }

.welcome { max-width: 720px; margin: 24px auto; padding: 36px 40px; }
.welcome h2 { font-size: 20px; margin-bottom: 6px; }
.steps { list-style: none; padding: 0; margin: 22px 0 26px; display: grid; gap: 14px; }
.steps li { display: flex; gap: 14px; align-items: flex-start; }
.steps p { margin: 2px 0 0; color: var(--text-2); font-size: 13px; }
.num {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: color-mix(in srgb, var(--primary) 12%, transparent);
  color: var(--primary);
  font-weight: 600;
  font-size: 13px;
}
.offline { margin-top: 18px; color: var(--danger); font-size: 13px; }
</style>
