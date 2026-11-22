# 运行与操作

## 环境

- Python 3.12（建议）
- Windows 10/11（项目验证环境；Linux/macOS 理论可运行，但未验证）
- Streamlit、PyTorch、torchvision、NumPy、Pillow、pandas、OpenCV、Matplotlib

## 安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 准备权重

把 ImageNet ResNet18 权重放到：

```text
cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

SHA256：

```text
f37072fd47e89c5e827621c5baffa750081f7896bbacec160b1a16c560e07ec
```

程序不会自动下载权重；不要用随机初始化权重代替。

## 启动

```bash
streamlit run src/ui/app.py
```

打开 `http://127.0.0.1:8501`。

也可以双击 `scripts/run_app.bat`。

## 产品注册

1. 进入“产品注册”；
2. 填写产品名称/类别；
3. 上传正常参考图（至少 5 张）；
4. 上传独立校准图（建议 20–25 张，不能与参考图重复）；
5. 点击注册。

系统会生成 product_id、版本、manifest、bank 和阈值。

## 单图检测

1. 进入“产品检测”；
2. 选择产品；
3. 上传图片；
4. 点击“开始产品检测”。

输出：正常/异常、图像分数、阈值、版本、热图、叠加图、mask、结果 JSON。

## 批量检测

1. 进入“产品批量”；
2. 选择产品；
3. 上传多张图片；
4. 启动批次，等待完成；
5. 在“产品批次历史”查看结果，下载 CSV/JSON/ZIP。

单张损坏/空文件只标记当前项失败，不会中断其他图片。

## 漂移监测

1. 进入“产品漂移”；
2. 选择产品；
3. 上传至少 5 张同产品窗口图；
4. 点击“执行漂移分析”。

输出 `normal` / `observation` / `drift_warning` / `insufficient` / `failed`、原因和统计；预警不等于设备故障。

## 人工审核与更新

1. 在漂移/检测/批次结果中提交反馈；
2. 进入“人工审核”，查看原图、分数、阈值和漂移原因；
3. 选择确认正常、确认异常、环境变化或拒绝；
4. 只有 confirmed 正常/异常进入候选；
5. 选择候选、上传独立验证正常/异常样本；
6. 创建并验证新版本；
7. 只有 verdict=pass 后，人工点击“确认激活新版本”；
8. 验证失败/不充分保留旧 current。

## 数据目录

```text
cache/product_batches/   批量记录
cache/product_drift/     漂移窗口
products/<product_id>/   产品/版本/记忆/反馈/更新
```

正式使用时，通过环境变量指定产品根：

```powershell
$env:MICRODEFECT_PRODUCT_ROOT='D:\your_data\products'
```

## 常见问题

- **找不到权重**：把 `resnet18-f37072fd.pth` 放到 `cache/torch/hub/checkpoints/`。
- **产品不可检测**：检查产品 status、current_version、manifest、阈值和 bank 完整性。
- **漂移不可用**：旧产品没有持久化 source_samples 时无法建立基线，需要重新注册。
- **验证不可用**：确认正常/异常验证样本均 ≥5，且与 reference/calibration/update 样本 SHA 不重复。
- **更新未激活**：只有 pass 且人工点击才切换 current；current 变化会拒绝旧更新。
