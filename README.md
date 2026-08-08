# AR 游戏创建平台 · 路线A Demo

摄像头识别人体动作，实时驱动 VRM 游戏角色。**纯浏览器、零安装**，手机/电脑浏览器打开即可玩。

> 技术路线 A：MediaPipe Pose Landmarker（姿态识别）+ Three.js（渲染）+ three-vrm（VRM 模型驱动）+ 自研骨骼重定向。
> 对应调研文档：`docs/技术选型调研.md`

## 运行

```bash
npm install        # 安装依赖
npm run dev        # 启动开发服务器，默认 http://localhost:5173
```

打开后点击「开启摄像头」，浏览器请求权限时选择允许。站在摄像头前、全身入镜、双臂自然张开，角色会跟随你的动作。右上角小窗显示摄像头画面与检测到的关键点，左下角显示 FPS。

> 注意：浏览器要求摄像头必须在 `https` 或 `localhost` 下才能访问。本地 `npm run dev` 是 localhost，可直接用；部署到公网需配 HTTPS。

## 项目结构

```
index.html            页面与 UI（摄像头小窗、状态条、按钮）
src/main.ts           主入口：串联摄像头、姿态检测、重定向、渲染循环
src/pose.ts           封装 MediaPipe Pose Landmarker（逐帧检测，输出 33 个 3D 关键点）
src/retarget.ts       核心：BlazePose 33 关键点 → VRM Humanoid 骨骼重定向 + 低通滤波
src/scene.ts          Three.js 场景：渲染器、相机、灯光、地面、VRM 模型加载
docs/技术选型调研.md  三条技术路线对比与选型依据
```

## 工作原理

```
摄像头(WebRTC) → Pose Landmarker 33 关键点(world 3D) → 骨骼重定向 → VRM 角色驱动
```

- **识别层**：`Pose Landmarker` 输出 33 个 worldLandmarks（米制 3D 坐标，原点在髋中心）。
- **重定向层**：`Retargeter` 对每段骨骼取两个关键点的世界方向向量，转换到该骨骼父节点的本地坐标系，用 `setFromUnitVectors` 把静止本地方向对齐到目标方向，写入骨骼本地旋转，再做 slerp 低通滤波抗抖。
- **渲染层**：`Stage` 用 three-vrm 加载并渲染 VRM 模型。

三层解耦——识别出的是标准骨骼数据，未来把 `Pose Landmarker` 换成 3D 关键点模型（ThreeDPoseTracker 思路）或 ARKit 串流，平台渲染侧无需改动。

## 已知局限（Demo 阶段）

- 单目 2D→3D 深度歧义：侧身、大幅转身时关节会有跳动；当前重定向未驱动整体朝向，转身表现为肢体方向变化而非角色转身。
- 手指级精度未做（BlazePose 无手指，需要可叠加 Hand Landmarker）。
- 示例 VRM 模型与 MediaPipe wasm/模型在运行时从 CDN 加载，首次需联网。
- 精度约为「可玩 demo」级别，未达到产品级动作动捕。

## 进阶方向

- 识别层替换为 ThreeDPoseTracker 式 3D 关键点模型（Web 端用 ONNX Runtime Web 跑），消除深度歧义。
- 接入行为识别：把姿态序列映射为游戏操作（如「举手=跳跃」「蹲下=下蹲」）。
- 模型本地化：把 VRM 模型与 MediaPipe 资源放到 `public/`，支持离线运行与创作者自定义上传。
- 移动端原生 ARKit（路线 C）通过 VMC 协议把骨骼数据串流给本平台，作为高精度可选方案。
