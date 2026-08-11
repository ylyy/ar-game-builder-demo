import {
  FilesetResolver,
  HolisticLandmarker,
  type HolisticLandmarkerResult,
} from '@mediapipe/tasks-vision';

// wasm 运行时与模型全部本地化（public/），不依赖任何 CDN——
// jsdelivr 在国内/手机网络上经常拉不到，会导致检测静默全空（历史教训）。
const WASM_URL = import.meta.env.BASE_URL + 'wasm';
const MODEL_URL = import.meta.env.BASE_URL + 'models/holistic_landmarker.task';

/**
 * 封装 MediaPipe Holistic Landmarker 的初始化与逐帧检测。
 * 一次输出：pose 33 点（归一化 + 米制 world）、face 478 点、双手各 21 点，
 * 供 kalidokit 做 躯干/头部/表情/手指 的完整重定向。
 *
 * delegate 固定 CPU：与 kiarina 参考实现一致（未指定即 CPU），避免 WebGL/GPU
 * 在部分机器（尤其手机）上创建成功但推理静默返回空。Holistic 模型较大，
 * 若帧率不足可后续尝试 delegate: 'GPU' 对比。
 */
export class HolisticDetector {
  private landmarker: HolisticLandmarker;
  private lastVideoTime = -1;

  private constructor(landmarker: HolisticLandmarker) {
    this.landmarker = landmarker;
  }

  static async create(): Promise<HolisticDetector> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);

    const landmarker = await HolisticLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
      runningMode: 'VIDEO',
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minHandLandmarksConfidence: 0.5,
    });
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
