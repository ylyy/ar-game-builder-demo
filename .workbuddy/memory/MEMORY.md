# 项目长期记忆

## 项目身份
- **argamebuilder**：AR 游戏创建平台。核心玩法 = 玩家用手机/电脑摄像头做动作，实时驱动游戏内角色模型。
- 当前阶段：技术验证期，先跑通「识别→驱动」最小闭环。

## 技术决策（已锁定路线A 先实现）
- 路线A：Web 全栈 = MediaPipe **Holistic** Landmarker(JS) + Three.js + three-vrm（Vite + TS 工程）。2026-08-10 已从 Pose(33点) 升级到 Holistic（Pose 33 + FaceMesh 468 + Hands 21×2 + 52 blendshape）。
- **重定向算法（2026-08-11 起）**：从自研 kiarina 移植算法改为 **kalidokit 库**（移植自 vtube-sol/codevibersol-glitch 的 vrmAvatar.js）。识别层 = src/pose.ts 的 HolisticDetector（HolisticLandmarker），渲染层 = src/retarget.ts 用 `Kalidokit.Face/Pose/Hand.solve()` 三件套 → 欧拉角平滑写入 VRM 骨骼 + 表情 + lookAt。手/指/表情/头部由 kalidokit 内部处理，不再手写 basis。
- 架构铁律：**识别层与渲染层解耦**——识别输出标准骨骼数据（MotionFrame），平台渲染侧不感知识别方案。未来可把 MediaPipe 换成 3D 关键点模型或 ARKit 串流而不改渲染。
- 备选路线：B·Unity（Barracuda/VNect 3D 关键点，效果上限高但分发重）、C·移动原生（ARKit，精度最高但多端分裂）。

## 环境约定
- 使用 managed Node（/Users/test/.workbuddy/binaries/node/versions/22.22.2/bin/npm），联网安装正常。
- 摄像头仅 localhost/https 可用；部署公网需配 HTTPS。
- **部署**：GitHub Pages（仓库 `ylyy/ar-game-builder-demo`，地址 https://ylyy.github.io/ar-game-builder-demo/）。GitHub Actions 自动部署：push main 即 `npm ci → build → deploy-pages`。Vite `base: './'`（相对）、`VRM_URL` 用 `import.meta.env.BASE_URL` 拼接以适配子路径；`tsconfig` 需 `types:["vite/client"]`。

## 关键坐标约定（retarget.ts，勿再改错）
- **权威依据**：可运行的参考实现 kiarina/labs mediapipe-holistic-vrm（实测 173FPS）——`new Vector3(mirror ? x : -x, -y, -z)`。
- MediaPipe Pose **worldLandmarks 是世界坐标**（米制、原点在髋中心），**但 Y 轴实际朝下**（与图像归一化坐标同向）！之前按官方文档"Y 向上"改成不取反，导致角色上下颠倒（腿到头、头弯进胸口）。
- 到 VRM 的映射：**x 默认取反、y 取反、z 取反**（mirrorX/flipY 两个调试按钮可切换约定）。
- restDir 静止方向必须用**父骨骼**世界四元数转本地（与 update() 一致），不能用骨骼自身四元数。
- UI 兜底：镜像（翻转X）、翻转Y（调试）两个按钮。

## 重定向算法（2026-08-11 已换 kalidokit，以下为旧自研算法的历史结论，仅作参考勿再回退）
- 两个可运行参考：**kiarina/labs mediapipe-holistic-vrm**（Web，173FPS）与 **GanniPiece/MetU**（Unity）。均确认：**躯干必须用双轴 basis**，四肢单方向即可。
- 躯干（kiarina 算法）：左右髋×左右肩 4 点建正交 basis（x=右髋-左髋，y=肩中-髋中，z=x×y，再正交化 y），权重 hips 0.5 / spine 0.25 / chest 0.15 / upperChest 0.1（本模型多一根 upperChest，kiarina 原版 0.55/0.25/0.20 无 upperChest）。
- 四肢：单方向对齐 + 平滑——参考实现同样不解四肢 twist，勿再自作主张加轴。
- **头/颈（2026-08-10 修复头翻扣后）**：面部 4 点(234/454/10/152)建 basis，neck 0.35 / head 0.65；面部缺失时回退 Pose 鼻(0)+双耳(7/8) 兜底。勿再用「肩→耳」裸向量驱动 neck（BlazePose 耳部 world 关键点弱，会把头翻扣进胸口）。
- **手/指/表情（Holistic 升级后）**：手腕用手掌 3 点(腕0/食指根5/小指根17)建 basis + 110° 钳制防翻转（MAX_WRIST_ANGLE）；手指 5×3 节单方向对齐；表情 blendshape→VRM 预设（blinkLeft/Right、aa/ih/ou/ee/oh/happy/surprised）映射见 kiarina updateExpressions。
- **坐标系坑**：pose/双手用 worldLandmarks（米制），face 用 faceLandmarks（图像归一化）——两套坐标不要混用；face 的 basis 只取相对向量，归一化后可用。
- 早期"一直映射失败"的两大根因已修：①Y 取反错（上下颠倒）；②躯干无左右轴（转身/侧倾丢失）。
- MetU 坐标同样 x/y/z 全取反（SpineCalculator 里 `-X/-Y/-Z`），与我们的约定互相印证。

## kalidokit 方案要点（2026-08-11 起，勿再改回自研）
- 依赖：`kalidokit@1.1.5`（npm 已装），`kalidokit` 输出**弧度欧拉角**（THREE.Euler XYZ 序）直接写 `bone.rotation`，无自研四元数。
- API：`Kalidokit.Face.solve(faceLms,{runtime:'mediapipe',video})` → head/eye/mouth.shape/pupil；`Kalidokit.Pose.solve(worldLms, normLms,{runtime:'mediapipe',video,enableLegs:true})` → Hips/Spine/四肢；`Kalidokit.Hand.solve(handLms,'Left'|'Right')` → 手指（key 形如 `LeftWrist`/`LeftThumbProximal`…，需 `Wrist→Hand` 改名 + 首字母小写映射到 VRM 骨）。
- **video 参数必须传 HTMLVideoElement 或 null**（传 `{width,height}` 对象会取 `videoWidth` 变 NaN）；null 时用归一化坐标（mediapipe 模式本就归一化）。
- 头/颈：head 全量 + neck 取 head×0.3；表情 blinkLeft/Right ← eye.l/r 取反，aa/ih/ou/ee/oh ← mouth.shape A/I/O/U/E；眼球 pupil → vrm.lookAt.yaw/pitch。
- 坐标：kalidokit 已内置 mediapipe「画面侧↔VRM 骨骼侧」左右映射（11→leftUpperArm），与项目"镜像视频+不旋转 vrm.scene"天然兼容；mirrorX/flipY 按钮仅作为兜底（翻转归一化 x→1-x / world x→-x 再喂 solve）。
- 平滑：保留逐轴欧拉 lerp + deadzone(0.001) + NaN guard；**不驱动 hips.position**（整体位置由 scene.userOffset 拖拽控制，识别层只改骨骼旋转）。
- 已修复的坑：HolisticLandmarkerOptions **没有** numPoses/minTrackingConfidence/minHandPresenceConfidence 字段；VRM 骨骼**没有** ThumbIntermediate（kalidokit 会输出，映射后 getNormalizedBoneNode 返回 null 自动跳过，ALL_BONES 里也别写）。

## 待办/未决
- 单目深度歧义、转身不敏感——效果不足时升级到 3D 关键点识别或 ARKit。
- 行为识别映射（姿态序列→游戏操作）尚未做。模型/资源已本地化（public/models/：Seed-san.vrm + holistic_landmarker.task，13.6MB）。
