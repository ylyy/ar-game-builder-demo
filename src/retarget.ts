import * as THREE from 'three';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import type { PoseLandmarkerResult } from '@mediapipe/tasks-vision';

/**
 * BlazePose 33 关键点索引（本项目用到的）：
 * 0 nose | 7 left_ear 8 right_ear
 * 11 left_shoulder 12 right_shoulder
 * 13 left_elbow 14 right_elbow | 15 left_wrist 16 right_wrist
 * 23 left_hip 24 right_hip | 25 left_knee 26 right_knee | 27 left_ankle 28 right_ankle
 */

interface Seg {
  bone: VRMHumanBoneName;
  from: number[]; // 单值=该点，双值=两点中点
  to: number[];
}

// 每段骨骼由哪两个（组）关键点方向来驱动
const SEGMENTS: Seg[] = [
  { bone: 'spine', from: [23, 24], to: [11, 12] },
  { bone: 'chest', from: [11, 12], to: [7, 8] },
  { bone: 'upperChest', from: [11, 12], to: [7, 8] },
  { bone: 'neck', from: [11, 12], to: [7, 8] },
  { bone: 'head', from: [7, 8], to: [0] },
  { bone: 'leftUpperArm', from: [11], to: [13] },
  { bone: 'leftLowerArm', from: [13], to: [15] },
  { bone: 'rightUpperArm', from: [12], to: [14] },
  { bone: 'rightLowerArm', from: [14], to: [16] },
  { bone: 'leftUpperLeg', from: [23], to: [25] },
  { bone: 'leftLowerLeg', from: [25], to: [27] },
  { bone: 'rightUpperLeg', from: [24], to: [26] },
  { bone: 'rightLowerLeg', from: [26], to: [28] },
];

// 子骨骼链，用于预计算每个骨骼在静止姿态下的「子骨骼相对父骨骼」的本地方向
const BONE_CHAIN: Partial<Record<VRMHumanBoneName, VRMHumanBoneName>> = {
  spine: 'chest',
  chest: 'upperChest',
  upperChest: 'neck',
  neck: 'head',
  leftUpperArm: 'leftLowerArm',
  leftLowerArm: 'leftHand',
  rightUpperArm: 'rightLowerArm',
  rightLowerArm: 'rightHand',
  leftUpperLeg: 'leftLowerLeg',
  leftLowerLeg: 'leftFoot',
  rightUpperLeg: 'rightLowerLeg',
  rightLowerLeg: 'rightFoot',
};

/**
 * 把 MediaPipe 33 关键点（worldLandmarks）重定向到 VRM 人体骨骼。
 *
 * 算法：对每段骨骼，取两个关键点的「世界方向向量」，转换到该骨骼父节点
 * 的本地坐标系，再用 setFromUnitVectors 把静止本地方向对齐到目标方向，
 * 最终写入骨骼的本地旋转（local quaternion）。并做 slerp 低通滤波抗抖。
 *
 * 关键坐标约定（MediaPipe BlazePose GHUM worldLandmarks，米制 3D）：
 *   原点 = 髋中心；这是「世界坐标」（与图像归一化坐标不同）。
 *   权威实测参考（kiarina/labs mediapipe-holistic-vrm，可运行）的转换是：
 *     new Vector3(mirror ? x : -x, -y, -z)
 *   即默认 x、y、z 全部取反：
 *     x：取反（world x+ 对应镜像后的显示方向，默认按非镜像视图处理）；
 *     y：取反（worldLandmarks 的 Y 实际朝下，与 VRM +Y 向上相反！历史教训：
 *        之前按"文档 Y 向上"改成不取反，导致角色上下颠倒——腿到头、头弯进胸口）；
 *     z：取反（world z+ 远离相机，取反后朝向相机 = VRM +Z 正前方）。
 *   mirrorX / flipY 两个调试开关在两种约定间切换。
 */
export class Retargeter {
  private vrm: VRM;
  private restDir = new Map<VRMHumanBoneName, THREE.Vector3>();
  private smoothed = new Map<VRMHumanBoneName, THREE.Quaternion>();
  private readonly smoothing = 0.45;
  // 每帧旋转增量上限（弧度）：避免单帧噪声把身体猛地掰弯/抖一下
  private readonly maxStep = THREE.MathUtils.degToRad(22);
  // 单段骨骼所用关键点的最低平均 visibility，低于此值跳过该段（不更新姿势）
  private readonly minVisibility = 0.5;
  private mirrorX = false;
  /** 调试用：上下翻转（正常情况下应为 false，Y 轴是世界坐标、向上） */
  private flipY = false;
  private paused = false;
  private freezeUntil = 0;

  constructor(vrm: VRM, opts?: { mirrorX?: boolean; flipY?: boolean }) {
    this.vrm = vrm;
    if (opts?.mirrorX !== undefined) this.mirrorX = opts.mirrorX;
    if (opts?.flipY !== undefined) this.flipY = opts.flipY;

    vrm.scene.updateMatrixWorld(true);

    // 预计算静止本地方向：子骨骼相对父骨骼的位移方向（世界空间）→ 转「父骨骼本地系」
    // 注意：normalized bone 的 .position 是相对 hips 的全身坐标，不能直接当父本地系方向用，
    // 否则 rest 会被 Y 稀释，导致「水平类动作（弯肘/抬臂）方向控制力丢失」。
    // 关键：这里必须用「父骨骼」的世界四元数（骨骼本地系 = 父骨骼坐标系），
    // 与 update() 中 parent.getWorldQuaternion() 保持一致；之前误用骨骼自身四元数，
    // 在 T-pose 下手臂等骨骼自身有旋转时，rest 方向算错会导致肢体姿态明显不对。
    const bw = new THREE.Vector3();
    const cw = new THREE.Vector3();
    const bpq = new THREE.Quaternion();
    for (const [bone, child] of Object.entries(BONE_CHAIN)) {
      const bn = vrm.humanoid?.getNormalizedBoneNode(bone as VRMHumanBoneName);
      const cn = vrm.humanoid?.getNormalizedBoneNode(child as VRMHumanBoneName);
      if (bn && cn) {
        bn.getWorldPosition(bw);
        cn.getWorldPosition(cw);
        if (bn.parent) bpq.copy(bn.parent.getWorldQuaternion(new THREE.Quaternion()));
        else bpq.copy(bn.getWorldQuaternion(new THREE.Quaternion()));
        const localDir = cw.sub(bw).normalize().applyQuaternion(bpq.clone().invert());
        this.restDir.set(bone as VRMHumanBoneName, localDir);
      }
    }
    // head 无子骨骼，静止朝 +Y（相对 neck 向上）
    this.restDir.set('head', new THREE.Vector3(0, 1, 0));

    for (const seg of SEGMENTS) {
      const bn = vrm.humanoid?.getNormalizedBoneNode(seg.bone);
      if (bn) this.smoothed.set(seg.bone, bn.quaternion.clone());
    }
  }

  /** 切换水平镜像（设备/约定差异兜底用，UI 暴露按钮） */
  setMirrorX(on: boolean): void {
    if (this.mirrorX === on) return;
    this.mirrorX = on;
    this.resetSmoothed();
  }

  /** 调试用：切换上下翻转（正常情况下关；若设备/模型约定相反再开） */
  setFlipY(on: boolean): void {
    if (this.flipY === on) return;
    this.flipY = on;
    this.resetSmoothed();
  }

  private resetSmoothed(): void {
    for (const seg of SEGMENTS) {
      const bone = this.vrm.humanoid?.getNormalizedBoneNode(seg.bone);
      if (bone) this.smoothed.set(seg.bone, bone.quaternion.clone());
    }
  }

  private toVec(lm: { x: number; y: number; z: number; visibility?: number }[], idx: number): THREE.Vector3 {
    const p = lm[idx];
    // 对齐可运行的参考实现：默认 x/y/z 全取反（镜像/翻转Y 按钮可切换）
    //   x：默认取反（-p.x）；开启镜像后不取反（+p.x）
    //   y：默认取反（-p.y，worldLandmarks Y 朝下）；翻转Y 开启后不取反（+p.y）
    //   z：始终取反（world z+ 远离相机 → 朝相机 = VRM +Z 正前方）
    const x = this.mirrorX ? p.x : -p.x;
    const y = this.flipY ? p.y : -p.y;
    return new THREE.Vector3(x, y, -p.z);
  }

  private point(
    lm: { x: number; y: number; z: number; visibility?: number }[],
    idx: number[],
  ): THREE.Vector3 {
    if (idx.length === 1) return this.toVec(lm, idx[0]);
    return this.toVec(lm, idx[0]).add(this.toVec(lm, idx[1])).multiplyScalar(0.5);
  }

  update(result: PoseLandmarkerResult): boolean {
    // 暂停追踪期间保持当前姿态不变
    if (this.paused) return false;
    // reset() 之后冻结一段时间，避免被追踪立刻覆盖
    if (performance.now() < this.freezeUntil) return false;
    const world = result.worldLandmarks?.[0];
    if (!world || world.length < 33) return false;
    const humanoid = this.vrm.humanoid;
    if (!humanoid) return false;

    const parentQuat = new THREE.Quaternion();
    const tmpLocal = new THREE.Vector3();
    const targetQuat = new THREE.Quaternion();

    // 关键点 visibility 过滤：算一段用到所有关键点的平均 visibility，
    // 低于阈值就跳过（保持上一帧姿态），避免部分入镜/识别失败时把身体驱动成扭曲。
    const segVis = (idxs: number[]): number => {
      let s = 0; let n = 0;
      for (const i of idxs) {
        const l = world[i];
        if (l) { s += (l.visibility ?? 0); n++; }
      }
      return n ? s / n : 0;
    };

    for (const seg of SEGMENTS) {
      const bone = humanoid.getNormalizedBoneNode(seg.bone);
      const rest = this.restDir.get(seg.bone);
      if (!bone || !rest) continue;

      if (segVis([...seg.from, ...seg.to]) < this.minVisibility) continue;

      const from = this.point(world, seg.from);
      const to = this.point(world, seg.to);
      const worldDir = to.clone().sub(from);
      if (worldDir.lengthSq() < 1e-6) continue;
      worldDir.normalize();

      const parent = bone.parent;
      if (parent) parent.getWorldQuaternion(parentQuat);
      else parentQuat.identity();
      tmpLocal.copy(worldDir).applyQuaternion(parentQuat.clone().invert()).normalize();

      targetQuat.setFromUnitVectors(rest, tmpLocal);

      let sm = this.smoothed.get(seg.bone);
      if (!sm) {
        sm = new THREE.Quaternion();
        this.smoothed.set(seg.bone, sm);
      }
      // 先按平滑系数插值，再限制“单帧旋转增量”，避免首帧/噪声把身体掰弯
      const before = sm.clone();
      const after = before.clone().slerp(targetQuat, this.smoothing);
      const stepAngle = before.angleTo(after);
      if (stepAngle > this.maxStep) {
        sm.copy(before).slerp(targetQuat, this.maxStep / stepAngle);
      } else {
        sm.copy(after);
      }
      bone.quaternion.copy(sm);
    }
    return true;
  }

  /** 暂停追踪（保持当前骨骼姿态）；用于"重置姿态"后让角色停在 T-pose */
  pause(): void {
    this.paused = true;
  }

  /** 恢复追踪；恢复时清空平滑缓存，避免从反转态 slid 进入 */
  resume(): void {
    if (!this.paused) return;
    this.paused = false;
    this.resetSmoothed();
  }

  isPaused(): boolean {
    return this.paused;
  }

  /** 把所有驱动骨骼复位到 T-pose，并暂停追踪 → 角色会一直停在 T-pose 直到用户点"继续追踪" */
  reset(): void {
    for (const seg of SEGMENTS) {
      const bone = this.vrm.humanoid?.getNormalizedBoneNode(seg.bone);
      if (bone) bone.quaternion.identity();
      this.smoothed.set(seg.bone, new THREE.Quaternion());
    }
    this.vrm.humanoid?.getNormalizedBoneNode('hips')?.position.set(0, 0, 0);
    this.paused = true;
    this.freezeUntil = performance.now() + 1200; // 状态栏文案过渡
  }

  /**
   * 把骨骼清零到 T-pose，但**不暂停追踪**。
   * 用于 VRM 加载完成后立即调用：示例 VRM（VRM1_Constraint_Twist_Sample 等）的导出姿势
   * 通常不是 T-pose（为了展示 twist 约束会带非零旋转），如果不主动清零，
   * 启动后未开启摄像头时角色会停在"扭曲出厂姿势"，看起来像识别错位。
   */
  applyRestPose(): void {
    for (const seg of SEGMENTS) {
      const bone = this.vrm.humanoid?.getNormalizedBoneNode(seg.bone);
      if (bone) bone.quaternion.identity();
      this.smoothed.set(seg.bone, new THREE.Quaternion());
    }
    this.vrm.humanoid?.getNormalizedBoneNode('hips')?.position.set(0, 0, 0);
    // 立即更新一次 scene，让渲染显示 T-pose
    this.vrm.update(0);
  }

  /** 距离冻结结束的剩余毫秒（用于 UI 显示） */
  getFreezeRemaining(): number {
    return Math.max(0, this.freezeUntil - performance.now());
  }
}