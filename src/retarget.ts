import * as THREE from 'three';
import type { Category, Landmark, NormalizedLandmark } from '@mediapipe/tasks-vision';
import type { VRM, VRMHumanBoneName, VRMHumanoid } from '@pixiv/three-vrm';

/**
 * BlazePose 33 关键点索引（本项目用到的）：
 * 0 nose | 7 left_ear 8 right_ear
 * 11 left_shoulder 12 right_shoulder
 * 13 left_elbow 14 right_elbow | 15 left_wrist 16 right_wrist
 * 23 left_hip 24 right_hip | 25 left_knee 26 right_knee | 27 left_ankle 28 right_ankle
 */

/**
 * 一帧 Holistic 结果：识别层输出、渲染层消费（保持识别/渲染解耦）。
 * 注意坐标系：pose/双手 是 worldLandmarks（米制，原点髋中心）；face 是
 * faceLandmarks（图像归一化坐标）——与 kiarina 参考实现保持一致。
 */
export interface MotionFrame {
  pose: Landmark[];
  face: NormalizedLandmark[];
  leftHand: Landmark[];
  rightHand: Landmark[];
  blendshapes: Category[];
}

interface Seg {
  bone: VRMHumanBoneName;
  from: number[]; // 单值=该点，双值=两点中点
  to: number[];
}

// 四肢：单方向对齐（参考 kiarina 173fps 实现——同样不解四肢 twist）。
// 颈/头不走这里，由 updateHead() 单独驱动（面部 4 点主算法 + Pose 鼻耳兜底）。
const SEGMENTS: Seg[] = [
  { bone: 'leftUpperArm', from: [11], to: [13] },
  { bone: 'leftLowerArm', from: [13], to: [15] },
  { bone: 'rightUpperArm', from: [12], to: [14] },
  { bone: 'rightLowerArm', from: [14], to: [16] },
  { bone: 'leftUpperLeg', from: [23], to: [25] },
  { bone: 'leftLowerLeg', from: [25], to: [27] },
  { bone: 'rightUpperLeg', from: [24], to: [26] },
  { bone: 'rightLowerLeg', from: [26], to: [28] },
];

// 躯干：用「左右髋 × 左右肩」4 点建正交 basis，按权重分配各躯干骨（kiarina 同款）。
// 本模型多一根 upperChest，从 chest 里拆一部分给它。
const TORSO_BONES: [VRMHumanBoneName, number][] = [
  ['hips', 0.5],
  ['spine', 0.25],
  ['chest', 0.15],
  ['upperChest', 0.1],
];

// 五根手指的骨骼链与手部关键点索引（手 21 点）——kiarina 同款。
interface FingerChain {
  bones: string[];
  indices: number[];
}
const FINGERS: FingerChain[] = [
  { bones: ['ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal'], indices: [1, 2, 3, 4] },
  { bones: ['IndexProximal', 'IndexIntermediate', 'IndexDistal'], indices: [5, 6, 7, 8] },
  { bones: ['MiddleProximal', 'MiddleIntermediate', 'MiddleDistal'], indices: [9, 10, 11, 12] },
  { bones: ['RingProximal', 'RingIntermediate', 'RingDistal'], indices: [13, 14, 15, 16] },
  { bones: ['LittleProximal', 'LittleIntermediate', 'LittleDistal'], indices: [17, 18, 19, 20] },
];
const FINGER_BONE_NAMES: VRMHumanBoneName[] = [];
for (const side of ['left', 'right'] as const) {
  for (const f of FINGERS) for (const b of f.bones) FINGER_BONE_NAMES.push(`${side}${b}` as VRMHumanBoneName);
}

// 所有被驱动的骨骼（躯干 basis + 四肢 + 颈/头 + 手/指），用于 reset / applyRestPose / 平滑缓存
const ALL_BONES: VRMHumanBoneName[] = [
  ...TORSO_BONES.map(([b]) => b),
  ...SEGMENTS.map((s) => s.bone),
  'neck',
  'head',
  'leftHand',
  'rightHand',
  ...FINGER_BONE_NAMES,
];

// 手腕最大偏角：防止手掌翻转时手腕 360° 乱转（kiarina 同款钳制）
const MAX_WRIST_ANGLE = (110 * Math.PI) / 180;

/** 由手掌 3 点（腕 0 / 食指根 5 / 小指根 17）建正交 basis，得到手掌世界姿态 */
function palmBasis(
  wrist: THREE.Vector3,
  indexMcp: THREE.Vector3,
  littleMcp: THREE.Vector3,
): THREE.Quaternion | null {
  const across = indexMcp.clone().sub(littleMcp);
  const forward = indexMcp.clone().add(littleMcp).multiplyScalar(0.5).sub(wrist);
  if (across.lengthSq() < 1e-8 || forward.lengthSq() < 1e-8) return null;
  across.normalize();
  const normal = across.clone().cross(forward).normalize();
  if (normal.lengthSq() < 1e-8) return null;
  forward.copy(normal).cross(across).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(across, forward, normal));
}

/** 把旋转限制在距 identity 的最大角度内（保留旋转轴方向） */
function clampQuaternionAngle(q: THREE.Quaternion, maxAngle: number): THREE.Quaternion {
  const angle = q.angleTo(new THREE.Quaternion());
  if (angle <= maxAngle || angle < 1e-8) return q.clone();
  return new THREE.Quaternion().slerp(q, maxAngle / angle);
}

/**
 * 把 MediaPipe Holistic 关键点重定向到 VRM 人体骨骼。
 *
 * 坐标约定（MediaPipe worldLandmarks，米制 3D，原点=髋中心）：
 *   权威实测参考（kiarina/labs mediapipe-holistic-vrm，可运行）的转换是：
 *     new Vector3(mirror ? x : -x, -y, -z)
 *   即默认 x、y、z 全部取反：
 *     x：取反（world x+ 对应镜像后的显示方向）；
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
  // 手掌 rest 姿态（world）与 hand 骨骼 rest 世界四元数，用于手部重定向
  private restPalmBases = new Map<'left' | 'right', THREE.Quaternion>();
  private restWorldQuats = new Map<VRMHumanBoneName, THREE.Quaternion>();

  constructor(vrm: VRM, opts?: { mirrorX?: boolean; flipY?: boolean }) {
    this.vrm = vrm;
    if (opts?.mirrorX !== undefined) this.mirrorX = opts.mirrorX;
    if (opts?.flipY !== undefined) this.flipY = opts.flipY;

    vrm.scene.updateMatrixWorld(true);

    // 预计算静止本地方向：子骨骼相对父骨骼的位移方向（世界空间）→ 转「父骨骼本地系」。
    // 关键：必须用「父骨骼」的世界四元数（骨骼本地系 = 父骨骼坐标系），与 update() 一致；
    // 之前误用骨骼自身四元数，在 T-pose 下手臂等骨骼自身有旋转时，rest 方向算错导致肢体姿态明显不对。
    // 通用取法（kiarina findBoneChild）：BFS 找第一个带位移的子节点（手指末端没有子节点则跳过）。
    const bw = new THREE.Vector3();
    const cw = new THREE.Vector3();
    for (const name of ALL_BONES) {
      const bn = vrm.humanoid?.getNormalizedBoneNode(name);
      if (!bn) continue;
      const child = this.findBoneChild(bn);
      if (!child) continue;
      bn.getWorldPosition(bw);
      child.getWorldPosition(cw);
      const parent = bn.parent;
      const parentQ = parent ? parent.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion();
      const localDir = cw.sub(bw).normalize().applyQuaternion(parentQ.clone().invert());
      this.restDir.set(name, localDir);
    }

    // 手掌 rest basis + hand 骨骼 rest 世界四元数（供 updateHand 使用）
    for (const side of ['left', 'right'] as const) {
      this.capturePalmRest(side, vrm);
      const hb = vrm.humanoid?.getNormalizedBoneNode(`${side}Hand` as VRMHumanBoneName);
      if (hb) this.restWorldQuats.set(`${side}Hand` as VRMHumanBoneName, hb.getWorldQuaternion(new THREE.Quaternion()));
    }

    for (const bone of ALL_BONES) {
      const bn = vrm.humanoid?.getNormalizedBoneNode(bone);
      if (bn) this.smoothed.set(bone, bn.quaternion.clone());
    }
  }

  /** BFS 找第一个带位移的子节点（近似该骨骼的"伸长方向"） */
  private findBoneChild(node: THREE.Object3D): THREE.Object3D | null {
    const queue = [...node.children];
    while (queue.length > 0) {
      const child = queue.shift()!;
      if (child.position.lengthSq() > 1e-10) return child;
      queue.push(...child.children);
    }
    return null;
  }

  private capturePalmRest(side: 'left' | 'right', vrm: VRM): void {
    const hand = vrm.humanoid?.getNormalizedBoneNode(`${side}Hand` as VRMHumanBoneName);
    const index = vrm.humanoid?.getNormalizedBoneNode(`${side}IndexProximal` as VRMHumanBoneName);
    const little = vrm.humanoid?.getNormalizedBoneNode(`${side}LittleProximal` as VRMHumanBoneName);
    if (!hand || !index || !little) return;
    const basis = palmBasis(
      hand.getWorldPosition(new THREE.Vector3()),
      index.getWorldPosition(new THREE.Vector3()),
      little.getWorldPosition(new THREE.Vector3()),
    );
    if (basis) this.restPalmBases.set(side, basis);
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
    for (const bone of ALL_BONES) {
      const bn = this.vrm.humanoid?.getNormalizedBoneNode(bone);
      if (bn) this.smoothed.set(bone, bn.quaternion.clone());
    }
  }

  private toVec(lm: { x: number; y: number; z: number; visibility?: number }[], idx: number): THREE.Vector3 {
    const p = lm[idx];
    // 对齐可运行的参考实现：默认 x/y/z 全取反（镜像/翻转Y 按钮可切换）
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

  update(frame: MotionFrame | null): boolean {
    // 暂停追踪期间保持当前姿态不变
    if (this.paused) return false;
    // reset() 之后冻结一段时间，避免被追踪立刻覆盖
    if (performance.now() < this.freezeUntil) return false;
    if (!frame || !frame.pose || frame.pose.length < 33) return false;
    const world = frame.pose;
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

    // ============ 1) 躯干：左右髋×左右肩 4 点建正交 basis，按权重分配 ============
    if (
      segVis([11, 12, 23, 24]) >= this.minVisibility &&
      [11, 12, 23, 24].every((i) => world[i])
    ) {
      const leftHip = this.toVec(world, 23);
      const rightHip = this.toVec(world, 24);
      const leftShoulder = this.toVec(world, 11);
      const rightShoulder = this.toVec(world, 12);

      const hipCenter = leftHip.clone().add(rightHip).multiplyScalar(0.5);
      const shoulderCenter = leftShoulder.clone().add(rightShoulder).multiplyScalar(0.5);

      // 顺序与 kiarina 一致：先 x=右髋-左髋，再 y=肩-髋，z=x×y，再正交化
      const xAxis = rightHip.clone().sub(leftHip).normalize();
      const yAxis = shoulderCenter.clone().sub(hipCenter).normalize();
      let zAxis = new THREE.Vector3().crossVectors(xAxis, yAxis);
      if (zAxis.lengthSq() < 1e-8) {
        // 退化（如躺平/侧躺时髋肩共线）：回退到髋中点→肩中点的单方向
        zAxis.copy(yAxis.clone().cross(new THREE.Vector3(0, 0, 1)));
      }
      zAxis.normalize();
      // 重新正交化 y，保证 x/y/z 严格正交（kiarina 也这么做）
      const yOrth = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();

      const basis = new THREE.Matrix4().makeBasis(xAxis, yOrth, zAxis);
      const torsoWorld = new THREE.Quaternion().setFromRotationMatrix(basis);

      for (const [boneName, weight] of TORSO_BONES) {
        const bone = humanoid.getNormalizedBoneNode(boneName);
        if (!bone) continue;
        const parent = bone.parent;
        if (parent) parent.getWorldQuaternion(parentQuat);
        else parentQuat.identity();
        // 世界姿态按权重 slerp（从 identity 朝 torsoWorld 插值），再转父本地
        const distributed = new THREE.Quaternion().slerp(torsoWorld, weight);
        targetQuat.copy(parentQuat.clone().invert().multiply(distributed));

        this.applySmoothed(bone, boneName, targetQuat);
      }
    }

    // ============ 1.5) 颈/头：面部 4 点建正交 basis（面部缺失时 Pose 鼻耳兜底） ============
    this.updateHead(frame, humanoid);

    // ============ 2) 四肢：单方向对齐（参考实现同样不解四肢 twist） ============
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

      this.applySmoothed(bone, seg.bone, targetQuat);
    }

    // ============ 3) 手/指/表情：移植自 kiarina（Holistic 才有这四路数据） ============
    this.updateHand('left', frame.leftHand);
    this.updateHand('right', frame.rightHand);
    this.updateFingers('left', frame.leftHand);
    this.updateFingers('right', frame.rightHand);
    this.updateExpressions(frame.blendshapes);

    return true;
  }

  /**
   * 颈/头重定向（移植 kiarina 的面部 head basis）。
   * 主算法用面部 4 点：234/454 左右眼外角（x 轴）、10 额头/152 下巴（y 轴）建正交 basis，
   * 再按 neck 0.35 / head 0.65 分配（世界 slerp 后转父本地写入）。
   * 面部不可靠（<455 点）时回退到 Pose 鼻+双耳近似，保证头朝上不翻扣。
   *
   * ⚠️ 关键约定：zAxis 用 `yAxis.cross(xAxis)`（而非 kiarina 的 `xAxis.cross(yAxis)`），
   * 让 zAxis 朝向 +Z（指向相机）——VRM 模型在 three.js 默认相机下 T-pose 面朝 -Z，
   * 反向 z 会把头部绕 Y 转 180°、脸朝后（用户截图已确认此 bug）。躯干因为 T-pose 左右对称看不出来，头部因为有正反面立刻暴露。
   */
  private updateHead(frame: MotionFrame, humanoid: VRMHumanoid): void {
    const face = frame.face;
    if (face && face.length > 454) {
      const left = this.toVec(face, 234);
      const right = this.toVec(face, 454);
      const forehead = this.toVec(face, 10);
      const chin = this.toVec(face, 152);
      const xAxis = right.clone().sub(left).normalize();
      const yAxis = forehead.clone().sub(chin).normalize();
      // zAxis 朝向 +Z（指向相机），让头部基础朝向与 VRM 模型 T-pose 一致
      const zAxis = yAxis.clone().cross(xAxis).normalize();
      if (zAxis.lengthSq() < 1e-8) return;
      yAxis.copy(zAxis).cross(xAxis).normalize();
      const basis = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
      const headWorld = new THREE.Quaternion().setFromRotationMatrix(basis);
      this.applyHeadWeights(headWorld, humanoid);
      return;
    }
    this.updateHeadFromPose(frame.pose, humanoid);
  }

  /** 面部缺失时的兜底：用 Pose 的鼻(0)+双耳(7/8) 建头部 basis，保证头朝上 */
  private updateHeadFromPose(
    world: { x: number; y: number; z: number; visibility?: number }[],
    humanoid: VRMHumanoid,
  ): void {
    const need = [0, 7, 8, 11, 12, 23, 24];
    let vis = 0;
    let n = 0;
    for (const i of need) {
      const l = world[i];
      if (l) {
        vis += l.visibility ?? 0;
        n++;
      }
    }
    if (n === 0 || vis / n < this.minVisibility) return;

    const leftEar = this.toVec(world, 7);
    const rightEar = this.toVec(world, 8);
    const nose = this.toVec(world, 0);
    const earMid = leftEar.clone().add(rightEar).multiplyScalar(0.5);

    const xAxis = rightEar.clone().sub(leftEar);
    const fwd = nose.clone().sub(earMid);
    if (xAxis.lengthSq() < 1e-6 || fwd.lengthSq() < 1e-6) return;
    xAxis.normalize();
    fwd.normalize();

    // 取 fwd 去掉 x 分量后的竖直分量作为头部 up：保证默认朝上，点头时下倾
    const xComp = fwd.dot(xAxis);
    const yAxis = fwd.clone().sub(xAxis.clone().multiplyScalar(xComp)).normalize();
    if (yAxis.lengthSq() < 1e-6) return;
    // zAxis 朝向 +Z（指向相机），与 VRM 模型 T-pose 一致——勿再用 xAxis×yAxis（会得 -Z 让头转 180°）
    const zAxis = new THREE.Vector3().crossVectors(yAxis, xAxis).normalize();
    const yOrth = new THREE.Vector3().crossVectors(zAxis, xAxis).normalize();

    const basis = new THREE.Matrix4().makeBasis(xAxis, yOrth, zAxis);
    const headWorld = new THREE.Quaternion().setFromRotationMatrix(basis);
    this.applyHeadWeights(headWorld, humanoid);
  }

  /** 按 kiarina 权重把头部世界姿态分给 neck/head，转父本地写入 */
  private applyHeadWeights(headWorld: THREE.Quaternion, humanoid: VRMHumanoid): void {
    const parentQ = new THREE.Quaternion();
    const dist = new THREE.Quaternion();
    for (const [boneName, weight] of [
      ['neck', 0.35],
      ['head', 0.65],
    ] as Array<[VRMHumanBoneName, number]>) {
      const bone = humanoid.getNormalizedBoneNode(boneName);
      if (!bone) continue;
      const parent = bone.parent;
      if (parent) parent.getWorldQuaternion(parentQ);
      else parentQ.identity();
      dist.copy(parentQ).invert().multiply(new THREE.Quaternion().slerp(headWorld, weight));
      this.applySmoothed(bone, boneName, dist);
    }
  }

  /** 手腕/手掌：手掌 3 点 basis（0 腕 / 5 食指根 / 17 小指根），带 110° 钳制防翻转 */
  private updateHand(side: 'left' | 'right', points: Landmark[]): void {
    if (points.length < 18) return;
    const name = `${side}Hand` as VRMHumanBoneName;
    const node = this.vrm.humanoid?.getNormalizedBoneNode(name);
    const restWorld = this.restWorldQuats.get(name);
    const restBasis = this.restPalmBases.get(side);
    if (!node || !restWorld || !restBasis) return;

    const observed = palmBasis(this.toVec(points, 0), this.toVec(points, 5), this.toVec(points, 17));
    if (!observed) return;

    const desiredWorld = observed.multiply(restBasis.clone().invert()).multiply(restWorld);
    const parent = node.parent;
    const parentQ = parent ? parent.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion();
    const desiredLocal = parentQ.invert().multiply(desiredWorld);

    this.applySmoothed(node, name, clampQuaternionAngle(desiredLocal, MAX_WRIST_ANGLE));
  }

  /** 五根手指逐节对齐（手 21 点 world 坐标，与四肢同款单方向算法） */
  private updateFingers(side: 'left' | 'right', points: Landmark[]): void {
    if (points.length < 21) return;
    for (const finger of FINGERS) {
      for (let i = 0; i < finger.bones.length; i++) {
        const boneName = `${side}${finger.bones[i]}` as VRMHumanBoneName;
        const bone = this.vrm.humanoid?.getNormalizedBoneNode(boneName);
        const rest = this.restDir.get(boneName);
        if (!bone || !rest) continue;

        const from = this.toVec(points, finger.indices[i]);
        const to = this.toVec(points, finger.indices[i + 1]);
        const worldDir = to.clone().sub(from);
        if (worldDir.lengthSq() < 1e-6) continue;
        worldDir.normalize();

        const parent = bone.parent;
        const parentQ = parent ? parent.getWorldQuaternion(new THREE.Quaternion()) : new THREE.Quaternion();
        const targetLocal = worldDir.applyQuaternion(parentQ.clone().invert()).normalize();
        const targetQuat = new THREE.Quaternion().setFromUnitVectors(rest, targetLocal);

        this.applySmoothed(bone, boneName, targetQuat);
      }
    }
  }

  /** 面部 blendshape → VRM 表情（眨眼/张嘴/嘴型/笑脸…，kiarina 同款映射） */
  private updateExpressions(categories: Category[]): void {
    const manager = this.vrm?.expressionManager;
    if (!manager) return;
    const values = new Map(categories.map((c) => [c.categoryName, c.score]));
    const get = (name: string): number => values.get(name) ?? 0;
    const average = (l: string, r: string): number => (get(l) + get(r)) / 2;
    const targets: Record<string, number> = {
      blinkLeft: get('eyeBlinkLeft'),
      blinkRight: get('eyeBlinkRight'),
      aa: get('jawOpen'),
      ih: Math.max(average('mouthSmileLeft', 'mouthSmileRight') * 0.35, average('mouthStretchLeft', 'mouthStretchRight')),
      ou: get('mouthPucker'),
      ee: average('mouthStretchLeft', 'mouthStretchRight'),
      oh: get('mouthFunnel'),
      happy: average('mouthSmileLeft', 'mouthSmileRight') * 0.7,
      surprised: Math.max(average('eyeWideLeft', 'eyeWideRight'), get('jawOpen')) * 0.35,
    };
    for (const [name, target] of Object.entries(targets)) {
      const current = manager.getValue(name) ?? 0;
      manager.setValue(name, current + (target - current) * this.smoothing);
    }
  }

  /**
   * 平滑 + 单帧限幅后写入骨骼本地四元数。
   * 统一入口，躯干/四肢/头/手共用同一套抗抖逻辑。
   */
  private applySmoothed(
    bone: THREE.Object3D,
    key: VRMHumanBoneName,
    targetQuat: THREE.Quaternion,
  ): void {
    let sm = this.smoothed.get(key);
    if (!sm) {
      sm = new THREE.Quaternion();
      this.smoothed.set(key, sm);
    }
    // 先按平滑系数插值，再限制"单帧旋转增量"，避免首帧/噪声把身体掰弯
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
    for (const bone of ALL_BONES) {
      const bn = this.vrm.humanoid?.getNormalizedBoneNode(bone);
      if (bn) bn.quaternion.identity();
      this.smoothed.set(bone, new THREE.Quaternion());
    }
    this.vrm.humanoid?.getNormalizedBoneNode('hips')?.position.set(0, 0, 0);
    this.paused = true;
    this.freezeUntil = performance.now() + 1200; // 状态栏文案过渡
  }

  /**
   * 把骨骼清零到 T-pose，但**不暂停追踪**。
   * 用于 VRM 加载完成后立即调用：示例 VRM 的导出姿势通常不是 T-pose，
   * 如果不主动清零，启动后未开启摄像头时角色会停在"扭曲出厂姿势"。
   */
  applyRestPose(): void {
    for (const bone of ALL_BONES) {
      const bn = this.vrm.humanoid?.getNormalizedBoneNode(bone);
      if (bn) bn.quaternion.identity();
      this.smoothed.set(bone, new THREE.Quaternion());
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
