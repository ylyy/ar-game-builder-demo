import {
  FilesetResolver,
  PoseLandmarker,
  type PoseLandmarkerResult,
} from '@mediapipe/tasks-vision';

// MediaPipe wasm 运行时与姿态模型（运行时按需从 CDN 拉取）
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';

/**
 * 封装 MediaPipe Pose Landmarker 的初始化与逐帧检测。
 * 输出 33 个 worldLandmarks（米制 3D 坐标，原点在髋中心）。
 */
export class PoseDetector {
  private landmarker: PoseLandmarker;
  private lastVideoTime = -1;

  private constructor(landmarker: PoseLandmarker) {
    this.landmarker = landmarker;
  }

  static async create(): Promise<PoseDetector> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);

    const base = {
      runningMode: 'VIDEO' as const,
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    };

    let landmarker: PoseLandmarker;
    try {
      landmarker = await PoseLandmarker.createFromOptions(fileset, {
        ...base,
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      });
    } catch {
      // 部分设备 WebGL delegate 不可用，回退 CPU
      landmarker = await PoseLandmarker.createFromOptions(fileset, {
        ...base,
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
      });
    }
    return new PoseDetector(landmarker);
  }

  /** 对当前视频帧做一次检测；无新帧或视频未就绪时返回 null */
  detect(video: HTMLVideoElement): PoseLandmarkerResult | null {
    if (video.readyState < 2) return null;
    if (video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = video.currentTime;
    return this.landmarker.detectForVideo(video, performance.now());
  }

  close(): void {
    this.landmarker.close();
  }
}
