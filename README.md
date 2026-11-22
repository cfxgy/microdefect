# 微缺智学 —— 工业缺陷自适应检测系统

面向多品种、小批量工业视觉检测的研究原型：用少量正常样本建立产品独立记忆，提供图像级缺陷检测、批量检测、环境漂移监测、人工审核与受控版本更新。

> 这是一个可运行的软件原型，不是已经量产部署的工业设备。仓库不包含数据集、模型权重、产品记忆库或比赛材料。

![系统架构](docs/images/architecture.png)

## 解决的问题

工业现场常见的问题是：产品型号多、换型频繁；新产品往往只有正常件、缺陷样本少；光源老化、相机位移、角度和批次变化会让原有检测阈值失准；不同产品共用模型还可能互相干扰。

本项目的思路不是重新训练一个通用缺陷分类器，而是：

1. 用冻结的 ImageNet ResNet18 提取图像特征；
2. 为每个产品保存独立的正常特征记忆和阈值；
3. 用余弦相似度判断图像是否偏离正常；
4. 用窗口统计监测全局/空间特征、亮度和对比度漂移；
5. 人工审核反馈，只有确认正常/异常的反馈进入候选；
6. 候选版本通过独立验证后，由人工显式激活，旧版本保留。

## 主要功能

- 少样本产品注册与独立特征记忆
- 单张图片检测（分数、阈值、热图、叠加图、mask）
- 批量检测、失败隔离、批次历史、CSV/JSON/ZIP
- 产品级漂移监测与 observation/drift_warning
- 人工审核：确认正常、确认异常、环境变化、拒绝
- 候选版本、固定验证集、旧/新版本比较
- 独立验证通过后人工显式激活
- 多产品记忆与数据隔离

## 技术路线

```text
图片输入
  → 冻结 ResNet18 特征（全局 512 / 空间 256×14×14）
  → 产品独立 reference bank
  → 余弦相似度（1 − max cosine）
  → 产品阈值判断
  → 单图 / 批量 / 批次历史
  → 窗口漂移监测
  → 人工审核反馈
  → 候选版本
  → 固定验证集比较
  → 人工激活 / 失败保留旧版本
```

详细说明见：

- [架构与模块](docs/ARCHITECTURE.md)
- [实验与测试](docs/EXPERIMENTS.md)
- [运行操作](docs/OPERATIONS.md)
- [边界与限制](docs/LIMITATIONS.md)
- [展示图说明](docs/IMAGES.md)

## 仓库结构

```text
src/
  core/          产品、注册、检测、批量、漂移、反馈/更新
  anomaly/       余弦评分
  features/      冻结特征提取
  ui/            Streamlit 页面与服务层
tests/           自动化测试
scripts/         运行/注册/检测辅助脚本
docs/            文档与展示图
requirements.txt
LICENSE          Apache-2.0
```

## 快速开始

要求 Python 3.12（建议）。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

streamlit run src/ui/app.py
```

然后浏览器打开 `http://127.0.0.1:8501`。

也可以使用：

```bat
scripts\run_app.bat
```

## 模型权重

本仓库不包含预训练权重，也不会在运行时自动下载。

系统使用 ImageNet ResNet18 官方权重：

```text
文件名：resnet18-f37072fd.pth
SHA256：f37072fd47e89c5e827621c5baffa750081f7896bbacec160b1a16c560e07ec
```

放置位置（二选一）：

```text
cache/torch/hub/checkpoints/resnet18-f37072fd.pth
~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

权重需由使用者自行从可信来源获取，并确认其许可；不要用随机初始化权重替代。

## 数据

仓库不附带 MVTec AD、NEU-CLS 或任何产品图片。请使用有合法授权的数据，并按产品准备：

- 正常参考图：至少 5 张
- 独立校准图：建议 20–25 张，不能与参考图重复
- 独立测试图：用于人工检查，不参与注册

数据目录建议：

```text
datasets/
  your_product/
    reference/
    calibration/
    test_normal/
    test_defect/
```

`datasets/` 已被 `.gitignore` 排除。

## 测试

```bash
python -m unittest discover -s tests -p "test_*.py"
```

测试覆盖产品存储/版本、注册、检测、批量、漂移、反馈/更新、UI 等。部分测试需要本地权重或样本；没有数据/权重时会按测试设计跳过或失败，请先按上文准备。

## 已验证结果（摘要）

- MVTec AD 官方划分，15 类 × 5 seed × 5-shot：
  - Image AUROC **0.8494**，Image AP **0.9401**，Image F1 **0.7605**
  - Pixel AUROC **0.9235**，Pixel AP **0.3825**
- MVTec AD Group-disjoint，4 类（bottle/pill/toothbrush/zipper）：
  - Image AUROC **0.8460**，Image AP **0.9132**，Image F1 **0.6517**
  - 仅 4 类、单 seed，不能外推到 15 类，也不能与官方划分直接比较优劣
- 产品级演示：A 正常图 0.012513 / 异常图 0.048478 / 阈值 0.021150
- 漂移演示：亮度×1.5 窗口 → `drift_warning`
- 全量测试：**504 项 / 503 通过 / 1 跳过 / 0 失败**

结果依赖数据划分、阈值和样本数量，**不是生产级误报率/漏报率保证**。详见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 项目边界

- 当前通过图片上传检测，不是实时产线连续检测；
- 不控制机械臂，不自动调整工业相机；
- 没有 PLC/MES/QMS 接口；
- 漂移阈值来自校准基线，属于实验性设置；
- 产品级验证覆盖有限，需要真实生产环境和更多类别验证；
- 3D 图为概念设计效果图，不是实物照片或部署证据。

## 许可

代码采用仓库根目录的 **Apache License 2.0**。数据集、预训练权重和 3D 建模图不自动包含在本许可范围内，使用前请分别确认其授权。
