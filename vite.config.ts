import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

export default defineConfig({
  // 相对 base：部署到 GitHub Pages 子路径(ylyy.github.io/REPO)或根路径都能正确解析资源
  base: './',
  plugins: [basicSsl()],
  server: {
    host: true,
    port: 5173,
  },
  build: {
    target: 'es2020',
    outDir: 'dist',
  },
});
