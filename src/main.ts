import { Stage } from './scene';
import { PoseDetector } from './pose';
import { Retargeter } from './retarget';

const stage = new Stage(document.getElementById('stage')!);
const video = document.getElementById('video') as HTMLVideoElement;
const overlay = document.getElementById('overlay') as HTMLCanvasElement;
const statusEl = document.getElementById('status')!;
const statusText = document.getElementById('status-text')!;
const fpsEl = document.getElementById('fps')!;
const dbgX = document.getElementById('off-x');
const dbgY = document.getElementById('off-y');
const dbgZ = document.getElementById('off-z');
const btnStart = document.getElementById('btn-start') as HTMLButtonElement;
const btnVideo = document.getElementById('btn-video') as HTMLButtonElement;
const btnResetPos = document.getElementById('btn-resetpos') as HTMLButtonElement;
const btnCal = document.getElementById('btn-calibrate') as HTMLButtonElement;
const btnResume = document.getElementById('btn-resume') as HTMLButtonElement;
const btnMirror = document.getElementById('btn-mirror') as HTMLButtonElement;
const btnFlipY = document.getElementById('btn-flipy') as HTMLButtonElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;

function setStatus(text: string, cls: '' | 'ok' | 'load'): void {
  statusText.textContent = text;
  statusEl.className = cls;
}

// 当前驱动源：'idle' | 'webcam' | 'file'
let source: 'idle' | 'webcam' | 'file' = 'idle';

function stopWebcam(): void {
  if (video.srcObject) {
    (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
    video.srcObject = null;
  }
}

async function boot(): Promise<void> {
  btnStart.disabled = true;
  btnMirror.disabled = true;
  btnFlipY.disabled = true;
  btnResume.disabled = true;

  setStatus('加载角色模型…', 'load');
  const vrm = await stage.loadVRM();
  setStatus('加载姿态模型…', 'load');
  const detector = await PoseDetector.create();
  const retargeter = new Retargeter(vrm);
  // VRM 文件本身的导出姿势可能不是 T-pose（例如官方 VRM1_Constraint_Twist_Sample
  // 为了展示 twist 约束带非零旋转），加载完先强制清零到 T-pose，避免
  // 「没开摄像头也看到扭曲身体」这个 false alarm。
  retargeter.applyRestPose();
  setStatus('就绪：请开启摄像头或加载真人视频', '');
  btnStart.disabled = false;
  btnMirror.disabled = false;
  btnVideo.disabled = false;
  btnResetPos.disabled = false;
  btnFlipY.disabled = false;

  // ---- 加载视频文件（用户主动选择，不自动加载任何视频） ----
  function loadVideoFile(src: string, label: string): void {
    stopWebcam();
    source = 'file';
    video.srcObject = null;
    video.removeAttribute('src');
    video.src = src;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.play().catch(() => {
      /* 部分浏览器需用户手势，循环里会重试 */
    });
    retargeter.resume();
    setStatus(label, 'ok');
  }

  // ---- 按钮：开启摄像头 ----
  btnStart.addEventListener('click', async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      stopFileVideo();
      source = 'webcam';
      video.srcObject = stream;
      await video.play();
      setStatus('追踪中（摄像头）', 'ok');
      btnStart.disabled = true;
    } catch (e) {
      setStatus('摄像头权限被拒绝', '');
      console.error(e);
    }
  });

  // ---- 按钮：加载测试视频（文件选择） ----
  btnVideo.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const f = fileInput.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    loadVideoFile(url, '自定义视频追踪中');
  });

  function stopFileVideo(): void {
    if (source === 'file') {
      video.pause();
      video.removeAttribute('src');
      video.load();
      source = 'idle';
    }
  }

  // ---- 按钮：重置位置（把角色放回桌面、居中） ----
  btnResetPos.addEventListener('click', () => {
    stage.resetPosition();
    setStatus('已重置角色位置', 'ok');
  });

  // ---- 按钮：重置姿态（清零到 T-pose 并暂停追踪）----
  btnCal.addEventListener('click', () => {
    retargeter.reset();
    btnResume.disabled = false;
    setStatus('已重置为 T-pose（追踪暂停中，点击继续追踪恢复）', 'ok');
  });

  // ---- 按钮：继续追踪 ----
  btnResume.addEventListener('click', () => {
    retargeter.resume();
    btnResume.disabled = true;
    setStatus('追踪中', 'ok');
  });

  // ---- 按钮：镜像 ----
  btnMirror.addEventListener('click', () => {
    const cur = btnMirror.getAttribute('data-on') === '1';
    const next = !cur;
    btnMirror.setAttribute('data-on', next ? '1' : '0');
    btnMirror.textContent = next ? '镜像：已开启' : '镜像：已关闭';
    retargeter.setMirrorX(next);
  });

  // ---- 按钮：翻转Y（调试用兜底） ----
  btnFlipY.addEventListener('click', () => {
    const cur = btnFlipY.getAttribute('data-on') === '1';
    const next = !cur;
    btnFlipY.setAttribute('data-on', next ? '1' : '0');
    btnFlipY.textContent = next ? '翻转Y：已开启' : '翻转Y：已关闭';
    retargeter.setFlipY(next);
    setStatus(next ? '已翻转Y（调试）' : '已恢复Y（正常）', 'ok');
  });

  let lastFps = performance.now();
  let frames = 0;

  function loop(): void {
    requestAnimationFrame(loop);

    // 文件源：若被浏览器策略暂停，尝试重新播放
    if (source === 'file' && video.paused && video.src) {
      video.play().catch(() => {
        /* ignore */
      });
    }

    const canDetect = (source === 'webcam' && !!video.srcObject) || (source === 'file' && !!video.src);
    const res = canDetect ? detector.detect(video) : null;
    if (res && res.worldLandmarks && res.worldLandmarks.length && res.landmarks?.[0]) {
      const updated = retargeter.update(res);
      if (updated) drawOverlay(res.landmarks[0]);
    }
    stage.render();

    // 调试读数：角色偏移（面板元素可能不存在，绝不能让这里拖垮主循环）
    const o = stage.getOffset();
    if (dbgX) dbgX.textContent = o.x.toFixed(2);
    if (dbgY) dbgY.textContent = o.y.toFixed(2);
    if (dbgZ) dbgZ.textContent = o.z.toFixed(2);

    frames++;
    const now = performance.now();
    if (now - lastFps > 500) {
      fpsEl.textContent = `${Math.round((frames * 1000) / (now - lastFps))} FPS`;
      frames = 0;
      lastFps = now;
    }
  }
  loop();
}

/** 在右上角小窗里画出检测到的关键点，作为视觉反馈 */
function drawOverlay(landmarks: { x: number; y: number }[]): void {
  const w = overlay.clientWidth || video.clientWidth;
  const h = overlay.clientHeight || video.clientHeight;
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
  }
  const ctx = overlay.getContext('2d')!;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#3fb950';
  for (const p of landmarks) {
    ctx.fillRect(p.x * w - 2, p.y * h - 2, 4, 4);
  }
}

boot().catch((e) => {
  console.error(e);
  setStatus('初始化失败：' + (e as Error).message, '');
});
