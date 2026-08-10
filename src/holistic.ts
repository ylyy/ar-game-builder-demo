import {
  FilesetResolver,
  HolisticLandmarker,
  type HolisticLandmarkerResult,
} from '@mediapipe/tasks-vision';

// MediaPipe wasm 运行时（版本与安装的 tasks-vision 对齐，按需从 CDN 拉取）
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
// Holistic 模型本地化（public/models/，兼容 GitHub Pages 子路径）
const MODEL_URL = import.meta.env.BASE_URL + 'models/holistic_landmarker.task';

/**
 * 封装 MediaPipe Holistic Landmarker 的初始化与逐帧检测。
 * 一次输出四路数据：Pose 33 点(world) + FaceMesh 468 点(归一化) + 左右手各 21 点(world)
 * + 52 个面部 blendshape（眨眼/嘴型/表情），对应 kiarina/labs 参考实现的识别层。
 */
export class HolisticDetector {
  private landmarker: HolisticLandmarker;
  private lastVideoTime = -1;

  private constructor(landmarker: HolisticLandmarker) {
    this.landmarker = landmarker;
  }

  static async create(): Promise<HolisticDetector> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);

    const base = {
      runningMode: 'VIDEO' as const,
      outputFaceBlendshapes: true,
      outputPoseSegmentationMasks: false,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minHandLandmarksConfidence: 0.5,
    };

    let landmarker: HolisticLandmarker;
    try {
      landmarker = await HolisticLandmarker.createFromOptions(fileset, {
        ...base,
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      });
    } catch {
      // 部分设备 WebGL delegate 不可用，回退 CPU
      landmarker = await HolisticLandmarker.createFromOptions(fileset, {
        ...base,
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
      });
    }
    return new HolisticDetector(landmarker);
  }

  /** 对当前视频帧做一次检测；无新帧或视频未就绪时返回 null */
  detect(video: HTMLVideoElement): HolisticLandmarkerResult | null {
    if (video.readyState < 2) return null;
    if (video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = video.currentTime;
    return this.landmarker.detectForVideo(video, performance.now());
  }

  close(): void {
    this.landmarker.close();
  }
}
