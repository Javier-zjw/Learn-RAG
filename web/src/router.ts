import { createRouter, createWebHistory } from 'vue-router'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: () => import('./views/HomeView.vue') },
    { path: '/new', component: () => import('./views/CreateView.vue') },
    { path: '/kb/:id', component: () => import('./views/KbView.vue') },
    // 文档 id 是带子目录的相对路径，用 (.*) 接住斜杠
    { path: '/kb/:id/doc/:docId(.*)', component: () => import('./views/DocView.vue') },
    { path: '/env', component: () => import('./views/EnvView.vue') },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
  scrollBehavior: () => ({ top: 0 }),
})
