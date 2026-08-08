import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, type VRM } from '@pixiv/three-vrm';

// 示例 VRM 模型：已下载到 public/models/ 本地加载（three-vrm 官方示例
// VRM1_Constraint_Twist_Sample）。替换为你自己的模型时，放到 public/models/
// 下并改这里的路径即可。
// 用 import.meta.env.BASE_URL 拼接，兼容 GitHub Pages 等子路径部署（base: './'）
export const VRM_URL =
  import.meta.env.BASE_URL + 'models/VRM1_Constraint_Twist_Sample.vrm';

/** Three.js 场景：渲染器、相机、灯光、地面，并负责加载与渲染 VRM 模型 */
export class Stage {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  vrm: VRM | null = null;

  /**
   * 用户拖拽产生的角色整体偏移（世界坐标）。用于在调试时把角色往上拖到桌面 /
   * 左右居中，而不依赖识别结果。识别层只改骨骼旋转，不改这个偏移，二者互不干扰。
   */
  readonly userOffset = new THREE.Vector3(0, 0, 0);
  /** 模型脚底相对原点的高度：默认让角色脚底正好落在网格(桌面)上，避免“下沉”。 */
  private groundY = 0;

  private last = performance.now();

  // ---- 鼠标拖拽角色相关 ----
  private raycaster = new THREE.Raycaster();
  private ndc = new THREE.Vector2();
  private dragPlane = new THREE.Plane();
  private dragPoint = new THREE.Vector3();
  private grabStart = new THREE.Vector3();
  private startOffset = new THREE.Vector3();
  private dragging = false;

  constructor(container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.touchAction = 'none';
    this.renderer.domElement.style.cursor = 'grab';

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      35,
      container.clientWidth / container.clientHeight,
      0.1,
      50,
    );
    this.camera.position.set(0, 1.2, 3.2);
    this.camera.lookAt(0, 1.0, 0);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1.1);
    hemi.position.set(0, 5, 0);
    this.scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(2, 4, 3);
    this.scene.add(dir);

    const grid = new THREE.GridHelper(10, 20, 0x3a4250, 0x232a33);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    this.scene.add(grid);

    this.attachDrag(this.renderer.domElement);
    window.addEventListener('resize', () => this.resize(container));
  }

  private resize(container: HTMLElement): void {
    const w = container.clientWidth;
    const h = container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  /** 在画布上拖拽 = 沿“面向相机的平面”平移角色（屏幕 X/Y + 深度 Z） */
  private attachDrag(canvas: HTMLCanvasElement): void {
    const toNdc = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      this.ndc.x = ((e.clientX - r.left) / r.width) * 2 - 1;
      this.ndc.y = -((e.clientY - r.top) / r.height) * 2 + 1;
    };

    canvas.addEventListener('pointerdown', (e) => {
      if (!this.vrm) return;
      toNdc(e);
      this.raycaster.setFromCamera(this.ndc, this.camera);
      const camDir = new THREE.Vector3();
      this.camera.getWorldDirection(camDir);
      // 过角色当前位置、法线=相机朝向的平面
      this.dragPlane.setFromNormalAndCoplanarPoint(camDir, this.vrm.scene.position);
      if (this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPoint)) {
        this.dragging = true;
        this.grabStart.copy(this.dragPoint);
        this.startOffset.copy(this.userOffset);
        canvas.setPointerCapture(e.pointerId);
        canvas.style.cursor = 'grabbing';
      }
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!this.dragging) return;
      toNdc(e);
      this.raycaster.setFromCamera(this.ndc, this.camera);
      if (this.raycaster.ray.intersectPlane(this.dragPlane, this.dragPoint)) {
        const dx = this.dragPoint.x - this.grabStart.x;
        const dy = this.dragPoint.y - this.grabStart.y;
        const dz = this.dragPoint.z - this.grabStart.z;
        this.userOffset.set(
          THREE.MathUtils.clamp(this.startOffset.x + dx, -3, 3),
          THREE.MathUtils.clamp(this.startOffset.y + dy, this.groundY - 1.5, this.groundY + 3),
          THREE.MathUtils.clamp(this.startOffset.z + dz, -3, 3),
        );
      }
    });

    const end = (e: PointerEvent) => {
      if (!this.dragging) return;
      this.dragging = false;
      canvas.style.cursor = 'grab';
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
  }

  /** 把角色放回默认位置（脚底落在网格/桌面上、居中） */
  resetPosition(): void {
    this.userOffset.set(0, this.groundY, 0);
  }

  /** 当前角色偏移，供调试读数与 UI 显示 */
  getOffset(): { x: number; y: number; z: number } {
    return { x: this.userOffset.x, y: this.userOffset.y, z: this.userOffset.z };
  }

  async loadVRM(onProgress?: (p: number) => void): Promise<VRM> {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync(VRM_URL, (e) => {
      onProgress?.(e.total ? e.loaded / e.total : 0);
    });
    const vrm = gltf.userData.vrm as VRM;
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.removeUnnecessaryJoints(gltf.scene);
    this.scene.add(vrm.scene);

    // 计算脚底高度，让角色默认就站在桌面(网格 y=0)上，避免“下沉”
    const box = new THREE.Box3().setFromObject(vrm.scene);
    this.groundY = -box.min.y;
    this.userOffset.set(0, this.groundY, 0);
    vrm.scene.position.copy(this.userOffset);

    this.vrm = vrm;
    return vrm;
  }

  render(): void {
    const now = performance.now();
    const dt = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    if (this.vrm) {
      this.vrm.update(dt);
      // 每帧应用用户拖拽偏移（识别层改的是骨骼旋转，这里改的是整体位置，互不覆盖）
      this.vrm.scene.position.copy(this.userOffset);
    }
    this.renderer.render(this.scene, this.camera);
  }
}
