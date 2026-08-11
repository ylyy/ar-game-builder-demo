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
 * 一次输出四路数据：Pose 33 点(world) + FaceMesh 468 点(归一化) + 左右手各 21 点(world)
 * + 52 个面部 blendshape（眨眼/嘴型/表情），对应 kiarina/labs 参考实现的识别层。
 *
 * 注意 delegate 选择：**固定用 CPU**（与 kiarina 参考实现一致，其未指定 delegate 即 CPU）。
 * Holistic 模型体积大，WebGL/GPU delegate 在部分机器（尤其手机）上创建成功但推理时
 * 静默返回空结果，正是"识别不到关键点"的高频根因。CPU 在 Mac 实测 170+ FPS，足够用。
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
      outputFaceBlendshapes: true,
      outputPoseSegmentationMasks: false,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
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
