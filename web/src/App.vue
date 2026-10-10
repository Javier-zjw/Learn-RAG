<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'
import AppSidebar from './components/AppSidebar.vue'
import { loadEnv, refreshKbs } from './store'

// 深色模式跟随系统，Element Plus 的深色变量挂在 html.dark 上
const media = window.matchMedia('(prefers-color-scheme: dark)')
const applyTheme = () => document.documentElement.classList.toggle('dark', media.matches)
applyTheme()

onMounted(() => {
  media.addEventListener('change', applyTheme)
  refreshKbs()
  loadEnv()
})
onBeforeUnmount(() => media.removeEventListener('change', applyTheme))
</script>

<template>
  <div class="shell">
    <AppSidebar />
    <main class="main">
      <RouterView v-slot="{ Component, route }">
        <Transition name="page" mode="out-in">
          <component :is="Component" :key="route.path" />
        </Transition>
      </RouterView>
    </main>
  </div>
</template>

<style scoped>
.shell {
  display: grid;
  grid-template-columns: 248px minmax(0, 1fr);
  height: 100%;
}
.main {
  overflow-y: auto;
  min-width: 0;
}
@media (max-width: 900px) {
  .shell { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
}
</style>
