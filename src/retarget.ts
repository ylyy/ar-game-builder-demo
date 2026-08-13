import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import type { HolisticLandmarkerResult } from '@mediapipe/tasks-vision';
import * as Kalidokit from 'kalidokit';
import type { TPose, TFace } from 'kalidokit';

const clamp01 = (v: number): number => Math.max(0, Math.min(1, v));

// landmark 的最小结构（x/y/z + 可选 visibility/score），兼容 tasks-vision 的
// NormalizedLandmark / Landmark
interface LandmarkLike {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

// 所有被 kalidokit 驱动的骨骼（躯干 + 头颈 + 四肢 + 手指），
// 用于 reset / applyRestPose / 平滑状态管理。
// 手指骨骼名与 kalidokit Hand 输出（LeftThumbProximal…）经
// 「首字母小写 + Wrist→Hand」转换后一一对应（见 _applyHand）。
const ALL_BONES: VRMHumanBoneName[] = [
  'hips', 'spine', 'chest', 'neck', 'head',
  'leftShoulder', 'rightShoulder',
  'leftUpperArm', 'rightUpperArm',
  'leftLowerArm', 'rightLowerArm',
  'leftHand', 'rightHand',
  'leftUpperLeg', 'rightUpperLeg',
  'leftLowerLeg', 'rightLowerLeg',
  'leftFoot', 'rightFoot',
  'leftThumbProximal', 'leftThumbDistal',
  'leftIndexProximal', 'leftIndexIntermediate', 'leftIndexDistal',
  'leftMiddleProximal', 'leftMiddleIntermediate', 'leftMiddleDistal',
  'leftRingProximal', 'leftRingIntermediate', 'leftRingDistal',
  'leftLittleProximal', 'leftLittleIntermediate', 'leftLittleDistal',
  'rightThumbProximal', 'rightThumbDistal',
  'rightIndexProximal', 'rightIndexIntermediate', 'rightIndexDistal',
  'rightMiddleProximal', 'rightMiddleIntermediate', 'rightMiddleDistal',
  'rightRingProximal', 'rightRingIntermediate', 'rightRingDistal',
  'rightLittleProximal', 'rightLittleIntermediate', 'rightLittleDistal',
];

/**
 * 用 kalidokit 算法把 MediaPipe Holistic 结果重定向到 VRM 人体骨骼。
 *
 * 算法来源：vtube-sol（codevibersol-glitch/vtube-sol）的 vrmAvatar.js——
 * 三件套 Kalidokit.Face/Pose/Hand.solve() 分别解算 头部/表情、躯干四肢、手指，
 * 输出弧度制欧拉角，经逐轴平滑（deadzone + NaN guard + 单帧限幅）写入骨骼。
 *
 * 坐标约定（与 vtube-sol / kalidokit 官方示例一致，勿改）：
 *   - 输入 landmarks 为 MediaPipe「原始帧」（未受 CSS 镜像影响），
 *     kalidokit 内部已处理「画面侧 ↔ VRM 骨骼侧」的左右映射，
 *     即 11(左肩)→leftUpperArm、12(右肩)→rightUpperArm。
 *   - VRM 角色面向相机（本项目相机在 +Z），与 kalidokit 默认朝向一致，
 *     因此无需旋转 vrm.scene；若个别模型/设备左右反了，用 UI 的镜像按钮兜底。
 *   - 头部欧拉角直接写 head，neck 取 head 的 0.3（与 vtube-sol 相同）。
 *   - 表情：blinkLeft/Right ← eye.l/r 取反，aa/ih/ou/ee/oh ← mouth.shape A/I/O/U/E。
 *   - 眼球追踪：pupil → vrm.lookAt.yaw/pitch（try/catch 保护，模型无 lookAt 时跳过）。
 *
 * 与 vtube-sol 的差异（有意保留）：
 *   - 不驱动 hips.position：整体位置由 scene 的 userOffset（拖拽）控制，
 *     识别层只改骨骼旋转（架构铁律），避免与拖拽定位冲突。
 *   - 保留原项目的 pause/reset/applyRestPose/mirrorX/flipY 调试接口。
 */
export class Retargeter {
  private vrm: VRM;
  private video: HTMLVideoElement | null = null;
  private mirrorX = false;
  /** 调试用：上下翻转（正常情况下应为 false） */
  private flipY = false;
  private paused = false;
  private freezeUntil = 0;

  // 平滑后的表达式值缓存（逐帧 lerp，避免表情抽搐）
  private exprCache = new Map<string, number>();
  // 平滑后的眼球角度缓存
  private lookYaw = 0;
  private lookPitch = 0;
  // 平滑后的髋部世界位移（vtube-sol 同款：身体重心跟随画面中的髋中心）
  private hipPos = new THREE.Vector3(0, 0, 0);

  // landmark 级时间平滑缓存（等效旧版 holistic 的 smoothLandmarks:true，
  // 新版 tasks-vision 无此选项，需自行实现，否则动作抖动不精确）
  private lmCache: {
    pose3d: LandmarkLike[] | null;
    pose2d: LandmarkLike[] | null;
    face: LandmarkLike[] | null;
    leftHand: LandmarkLike[] | null;
    rightHand: LandmarkLike[] | null;
  } = { pose3d: null, pose2d: null, face: null, leftHand: null, rightHand: null };

  constructor(vrm: VRM, opts?: { mirrorX?: boolean; flipY?: boolean }) {
    this.vrm = vrm;
    if (opts?.mirrorX !== undefined) this.mirrorX = opts.mirrorX;
    if (opts?.flipY !== undefined) this.flipY = opts.flipY;
  }

  /** kalidokit 需要 video 元素（用于按画面比例换算 hips 位置/眼睛） */
  setVideo(video: HTMLVideoElement | null): void {
    this.video = video;
  }

  /** 切换水平镜像（kalidokit 默认左右已正确；仅当设备/模型约定相反时开启） */
  setMirrorX(on: boolean): void {
    if (this.mirrorX === on) return;
    this.mirrorX = on;
  }

  /** 调试用：切换上下翻转（正常情况下关；若设备/模型约定相反再开） */
  setFlipY(on: boolean): void {
    if (this.flipY === on) return;
    this.flipY = on;
  }

  // ── 平滑写入 ───────────────────────────────────────────────────────────────

  private bone(name: VRMHumanBoneName): THREE.Object3D | null {
    return this.vrm.humanoid?.getNormalizedBoneNode(name) ?? null;
  }

  /**
   * 欧拉角平滑写入（vtube-sol 的 _lerp 原版逻辑 + NaN guard + 单帧限幅）。
   * deadzone 忽略亚阈值抖动；NaN 防护防止异常值污染骨骼。
   */
  private lerp(
    bone: THREE.Object3D | null,
    rot: { x?: number; y?: number; z?: number } | null | undefined,
    a = 0.15,
  ): void {
    if (!bone || !rot) return;
    const tx = rot.x ?? 0;
    const ty = rot.y ?? 0;
    const tz = rot.z ?? 0;
    if (!isFinite(tx) || !isFinite(ty) || !isFinite(tz)) return;
    const DEAD = 0.001;
    const step = (cur: number, target: number): number => {
      const d = target - cur;
      if (Math.abs(d) <= DEAD) return cur;
      return THREE.MathUtils.lerp(cur, target, a);
    };
    bone.rotation.x = step(bone.rotation.x, tx);
    bone.rotation.y = step(bone.rotation.y, ty);
    bone.rotation.z = step(bone.rotation.z, tz);
  }

  /** 表达式值平滑（vtube-sol 的 _lerpExpr） */
  private lerpExpr(
    exp: { setValue: (name: string, v: number) => void },
    name: string,
    target: number,
    a = 0.15,
  ): void {
    const prev = this.exprCache.get(name) ?? target;
    const val = THREE.MathUtils.lerp(prev, target, a);
    this.exprCache.set(name, val);
    try {
      exp.setValue(name, val);
    } catch {
      /* 模型无该预设时忽略 */
    }
  }

  // ── 主入口 ─────────────────────────────────────────────────────────────────

  /**
   * 对一帧 Holistic 结果做重定向。返回 true 表示本帧有可用的姿态数据
   * （供 UI 决定是否画关键点 overlay）。
   */
  update(result: HolisticLandmarkerResult): boolean {
    if (this.paused) return false;
    if (performance.now() < this.freezeUntil) return false;

    const pose2dRaw = result.poseLandmarks?.[0];
    const pose3dRaw = result.poseWorldLandmarks?.[0];
    const faceRaw = result.faceLandmarks?.[0];
    const leftHandRaw = result.leftHandLandmarks?.[0];
    const rightHandRaw = result.rightHandLandmarks?.[0];
    if (!pose2dRaw || !pose3dRaw) return false;
    if (!this.vrm.humanoid) return false;

    // landmark 级时间平滑（等效旧版 holistic smoothLandmarks:true，抗抖的关键）。
    // 每帧对原始 landmark 做指数平滑后生成新数组；kalidokit 的 solve 会原地修改
    // 传入数组（mediapipe 模式乘 imageSize），所以必须传副本。
    const smooth = (key: keyof Retargeter['lmCache'], lm: LandmarkLike[]): LandmarkLike[] => {
      const prev = this.lmCache[key];
      if (!prev || prev.length !== lm.length) {
        this.lmCache[key] = lm.map((p) => ({ ...p }));
        return this.lmCache[key] as LandmarkLike[];
      }
      const out = lm.map((p, i) => {
        const q = prev[i];
        return {
          x: q.x + (p.x - q.x) * 0.5,
          y: q.y + (p.y - q.y) * 0.5,
          z: q.z + (p.z - q.z) * 0.5,
          visibility: p.visibility,
        };
      });
      this.lmCache[key] = out;
      return out;
    };
    const pose3d = smooth('pose3d', pose3dRaw);
    const pose2d = smooth('pose2d', pose2dRaw);
    const face = faceRaw ? smooth('face', faceRaw) : null;
    const leftHand = leftHandRaw ? smooth('leftHand', leftHandRaw) : null;
    const rightHand = rightHandRaw ? smooth('rightHand', rightHandRaw) : null;

    // 镜像/翻转：仅在开关开启时翻转输入坐标（归一化 x→1-x，world x→-x）。
    // 注意：kalidokit 的 video 参数只接受元素或 null（传对象会取 videoWidth 变 NaN）；
    // 且 Face.solve 在 mediapipe 模式会原地把归一化坐标乘 imageSize.width，
    // 因此 video 未就绪（videoWidth=0）时必须传 null，否则 landmark 全乘 0 → 头部/表情全乱。
    // flip 始终返回新数组：防止 kalidokit 原地修改污染 lmCache（平滑缓存必须保持原始坐标）。
    const vid = this.video && this.video.videoWidth > 0 ? this.video : null;
    const flip = <T extends { x: number; y: number; z: number }>(lm: T[], normalized: boolean): T[] =>
      lm.map((p) => ({
        ...p,
        x: this.mirrorX ? (normalized ? 1 - p.x : -p.x) : p.x,
        y: this.flipY ? (normalized ? 1 - p.y : -p.y) : p.y,
      }));

    // 表情/头部
    if (face) {
      const rig = Kalidokit.Face.solve(flip(face, true), {
        runtime: 'mediapipe',
        video: vid,
      });
      if (rig) this.applyFace(rig);
    }

    // 躯干/四肢
    const rig = Kalidokit.Pose.solve(
      flip(pose3d, false),
      flip(pose2d, true),
      { runtime: 'mediapipe', video: vid, enableLegs: true },
    );
    if (rig) this.applyPose(rig, pose2d);

    // 手指（左右手分开解算；kalidokit 按 side 决定旋转方向）
    if (leftHand) {
      const rig = Kalidokit.Hand.solve(flip(leftHand, true), 'Left');
      if (rig) this.applyHand(rig);
    }
    if (rightHand) {
      const rig = Kalidokit.Hand.solve(flip(rightHand, true), 'Right');
      if (rig) this.applyHand(rig);
    }
    return true;
  }

  // ── Face：头/颈/表情/眨眼/眼球 ─────────────────────────────────────────────

  private applyFace(rig: TFace): void {
    // 头：kalidokit 输出弧度欧拉角，直接写 head；neck 取 head 的 0.3（同 vtube-sol）
    this.lerp(this.bone('head'), rig.head, 0.25);
    this.lerp(
      this.bone('neck'),
      {
        x: (rig.head?.x ?? 0) * 0.3,
        y: (rig.head?.y ?? 0) * 0.3,
        z: (rig.head?.z ?? 0) * 0.3,
      },
      0.25,
    );

    // 眼球 lookAt（模型无 lookAt 时静默跳过）
    if (rig.pupil) {
      try {
        if (this.vrm.lookAt) {
          this.lookYaw = THREE.MathUtils.lerp(this.lookYaw, -(rig.pupil.x ?? 0) * 15, 0.08);
          this.lookPitch = THREE.MathUtils.lerp(this.lookPitch, (rig.pupil.y ?? 0) * 10, 0.08);
          this.vrm.lookAt.yaw = this.lookYaw;
          this.vrm.lookAt.pitch = this.lookPitch;
        }
      } catch {
        /* ignore */
      }
    }

    const exp = this.vrm.expressionManager;
    if (!exp) return;

    // 眨眼：kalidokit 的 eye.l/r 是「睁眼程度」，取反给 VRM 的 blink
    this.lerpExpr(exp, 'blinkLeft', 1 - clamp01(rig.eye?.l ?? 1), 0.2);
    this.lerpExpr(exp, 'blinkRight', 1 - clamp01(rig.eye?.r ?? 1), 0.2);

    // 口型：A/I/O/U/E → aa/ih/ou/ee/oh（同 vtube-sol）
    const s = rig.mouth?.shape;
    if (s) {
      this.lerpExpr(exp, 'aa', clamp01(s.A ?? 0), 0.2);
      this.lerpExpr(exp, 'ih', clamp01(s.I ?? 0), 0.2);
      this.lerpExpr(exp, 'ou', clamp01(s.O ?? 0), 0.2);
      this.lerpExpr(exp, 'ee', clamp01(s.E ?? 0), 0.2);
      this.lerpExpr(exp, 'oh', clamp01(s.U ?? 0), 0.2);
    }
  }

  // ── Pose：躯干/四肢 ────────────────────────────────────────────────────────

  private applyPose(rig: TPose, lms: { visibility?: number }[]): void {
    const vis = (i: number): number => lms?.[i]?.visibility ?? 0;

    // 髋部：旋转 + 位置（vtube-sol 同款——身体重心跟随画面中的髋中心，
    // 这是"动作跟手"的关键：人往左偏角色左移、弯腰时躯干有位移感）
    const hips = this.bone('hips');
    if (hips && rig.Hips) {
      const p = rig.Hips.position;
      if (p) {
        // kalidokit 的 Hips.position 是画面归一化量级（x≈±0.6、z≈-1~0），
        // 直接照搬 vtube-sol 的映射（x 直取、z 取反、y 恒 0），
        // 但加 0.5 缩放避免角色位移过大冲出场景；y 保持 0——
        // 垂直定位由 scene.userOffset（拖拽）控制，不在这里抬升角色。
        const SCALE = 0.5;
        this.hipPos.x = THREE.MathUtils.lerp(this.hipPos.x, p.x * SCALE, 0.08);
        this.hipPos.z = THREE.MathUtils.lerp(this.hipPos.z, -p.z * SCALE, 0.08);
        hips.position.copy(this.hipPos);
      }
      if (rig.Hips.rotation) this.lerp(hips, rig.Hips.rotation, 0.07);
    }

    // 躯干：kalidokit 只给 Spine，spine/chest 都吃它（vtube-sol 的 ?? 兜底）
    this.lerp(this.bone('spine'), rig.Spine, 0.12);
    this.lerp(this.bone('chest'), rig.Spine, 0.12);

    // 肩/臂/腿：仅当端点关键点可见度足够时才更新（避免部分入镜时肢体扭曲）
    if (vis(11) > 0.5 && vis(13) > 0.4) this.lerp(this.bone('leftUpperArm'), rig.LeftUpperArm, 0.18);
    if (vis(13) > 0.4 && vis(15) > 0.3) this.lerp(this.bone('leftLowerArm'), rig.LeftLowerArm, 0.18);
    if (vis(12) > 0.5 && vis(14) > 0.4) this.lerp(this.bone('rightUpperArm'), rig.RightUpperArm, 0.18);
    if (vis(14) > 0.4 && vis(16) > 0.3) this.lerp(this.bone('rightLowerArm'), rig.RightLowerArm, 0.18);
    if (vis(23) > 0.5 && vis(25) > 0.4) this.lerp(this.bone('leftUpperLeg'), rig.LeftUpperLeg, 0.18);
    if (vis(25) > 0.4 && vis(27) > 0.3) this.lerp(this.bone('leftLowerLeg'), rig.LeftLowerLeg, 0.18);
    if (vis(24) > 0.5 && vis(26) > 0.4) this.lerp(this.bone('rightUpperLeg'), rig.RightUpperLeg, 0.18);
    if (vis(26) > 0.4 && vis(28) > 0.3) this.lerp(this.bone('rightLowerLeg'), rig.RightLowerLeg, 0.18);
  }

  // ── Hand：手指 ─────────────────────────────────────────────────────────────

  /** kalidokit 输出 key 如 LeftWrist/LeftThumbProximal…，转成 VRM 骨骼名 */
  private applyHand(rig: Record<string, { x?: number; y?: number; z?: number }>): void {
    for (const [key, rot] of Object.entries(rig)) {
      // kalidokit 命名 [Side]Wrist，但 VRM 骨骼是 [side]Hand；其余指节名一致
      const fixed = key.replace(/Wrist$/, 'Hand');
      const boneName = (fixed[0].toLowerCase() + fixed.slice(1)) as VRMHumanBoneName;
      const isWrist = /Hand$/.test(fixed);
      this.lerp(this.bone(boneName), rot, isWrist ? 0.15 : 0.25);
    }
  }

  // ── 状态管理（与原项目接口一致） ─────────────────────────────────────────────

  pause(): void {
    this.paused = true;
  }

  resume(): void {
    if (!this.paused) return;
    this.paused = false;
  }

  isPaused(): boolean {
    return this.paused;
  }

  /** 把所有驱动骨骼复位到 T-pose，并暂停追踪 → 角色停在 T-pose 直到"继续追踪" */
  reset(): void {
    this.resetBones();
    this.paused = true;
    this.freezeUntil = performance.now() + 1200;
  }

  /** 清零到 T-pose 但**不暂停**；VRM 加载完成后立即调用（出厂姿势常非 T-pose） */
  applyRestPose(): void {
    this.resetBones();
    this.vrm.update(0);
  }

  private resetBones(): void {
    for (const name of ALL_BONES) {
      const bn = this.bone(name);
      if (bn) bn.quaternion.identity();
    }
    this.vrm.humanoid?.getNormalizedBoneNode('hips')?.position.set(0, 0, 0);
    this.hipPos.set(0, 0, 0);
    // 清空 landmark 平滑缓存（避免从旧姿态"弹回"）
    this.lmCache = { pose3d: null, pose2d: null, face: null, leftHand: null, rightHand: null };
    // 表情清零
    this.exprCache.clear();
    const exp = this.vrm.expressionManager;
    if (exp) {
      try {
        const presets = ['blinkLeft', 'blinkRight', 'aa', 'ih', 'ou', 'ee', 'oh'];
        for (const p of presets) exp.setValue(p, 0);
      } catch {
        /* ignore */
      }
    }
  }

  /** 距离冻结结束的剩余毫秒（用于 UI 显示） */
  getFreezeRemaining(): number {
    return Math.max(0, this.freezeUntil - performance.now());
  }
}
