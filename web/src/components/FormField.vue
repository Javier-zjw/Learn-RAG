<script setup lang="ts">
// 表单里的一项：标签、控件、推荐理由。推荐值填入时整行闪一下，告诉用户哪些参数被改了
defineProps<{ label: string; unit?: string; reason?: string; flash?: boolean; hint?: string; error?: string }>()
</script>

<template>
  <div class="field" :class="{ flash }">
    <div class="label">
      <span>{{ label }}</span>
      <span v-if="unit" class="unit">{{ unit }}</span>
      <Transition name="fade">
        <el-tag v-if="reason" size="small" round effect="plain" class="rec">推荐</el-tag>
      </Transition>
    </div>
    <slot />
    <Transition name="slide">
      <div v-if="error" class="error">{{ error }}</div>
      <div v-else-if="reason" class="reason">{{ reason }}</div>
    </Transition>
    <div v-if="hint && !reason && !error" class="hint">{{ hint }}</div>
  </div>
</template>

<style scoped>
.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0 -10px;
  padding: 10px;
  border-radius: 8px;
}
.label { display: flex; align-items: center; gap: 6px; font-weight: 500; font-size: 13px; color: var(--text); }
.unit { font-weight: 400; color: var(--text-3); font-size: 12px; }
.rec { margin-left: auto; }
.reason, .error {
  font-size: 12px;
  line-height: 1.5;
  padding: 6px 9px;
  border-radius: 6px;
}
.reason { color: var(--text-2); background: color-mix(in srgb, var(--primary) 6%, transparent); }
.error { color: var(--danger); background: color-mix(in srgb, var(--danger) 8%, transparent); }
</style>
