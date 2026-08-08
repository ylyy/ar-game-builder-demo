# 项目长期记忆

## 项目身份
- **argamebuilder**：AR 游戏创建平台。核心玩法 = 玩家用手机/电脑摄像头做动作，实时驱动游戏内角色模型。
- 当前阶段：技术验证期，先跑通「识别→驱动」最小闭环。

## 技术决策（已锁定路线A 先实现）
- 路线A：Web 全栈 = MediaPipe Pose Landmarker(JS) + Three.js + three-vrm（Vite + TS 工程）
- 架构铁律：**识别层与渲染层解耦**——识别输出标准骨骼数据，平台渲染侧不感知识别方案。未来可把 MediaPipe 换成 3D 关键点模型或 ARKit 串流而不改渲染。
- 备选路线：B·Unity（Barracuda/VNect 3D 关键点，效果上限高但分发重）、C·移动原生（ARKit，精度最高但多端分裂）。

## 环境约定
- 使用 managed Node（/Users/test/.workbuddy/binaries/node/versions/22.22.2/bin/npm），联网安装正常。
- 摄像头仅 localhost/https 可用；部署公网需配 HTTPS。

## 关键坐标约定（retarget.ts，勿再改错）
- MediaPipe Pose **worldLandmarks 是世界坐标**（米制、原点在髋中心、Y 向上），与图像归一化坐标（Y 向下）不同。
- 到 VRM 的映射：**x 不取反**（person's right ≈ 图像右侧 ≈ VRM +X）、**y 不取反**（都向上）、**z 取反**（world z+ 远离相机 → 朝相机 = VRM +Z 正前方）。
- restDir 静止方向必须用**父骨骼**世界四元数转本地（与 update() 一致），不能用骨骼自身四元数。
- UI 兜底：镜像（翻转X）、翻转Y（调试）两个按钮。

## 待办/未决
- 单目深度歧义、转身不敏感、无手指精度——效果不足时升级到 3D 关键点识别或 ARKit。
- 行为识别映射（姿态序列→游戏操作）、模型/资源本地化（放 public/）尚未做。
