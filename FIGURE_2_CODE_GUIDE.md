# 图2与代码对应说明

## 1. 图2整体流程

论文图2展示 CFR 检测框架的整体结构。模型接收配准后的 CT 与 PET 图像，并按照以下流程完成病灶检测：

```text
CT image  ──> CT feature extractor  ──┐
                                       ├─> multi-scale TSAFF fusion ─> PAFPN ─> YOLOX head ─> detections
PET image ──> PET feature extractor ──┘
```

CT 和 PET 分别经过两个结构相同、参数独立的特征提取分支。两个分支产生三个尺度的特征，在每个尺度上通过 TSAFF 进行跨模态对齐与融合。融合特征随后进入自顶向下和自底向上的特征金字塔，最终由 YOLOX 检测头输出分类、目标置信度和边界框回归结果。

## 2. 图2与代码文件的对应关系

| 图2组成部分 | 主要代码位置 | 作用 |
| --- | --- | --- |
| CT/PET 双模态输入 | `train_tsaff.py`、`utils/dataloader.py` | 读取成对 CT/PET 图像并形成训练批次 |
| 双分支特征提取器 | `nets/darknet.py`、`nets/yolo.py` | 分别提取 CT 与 PET 的多尺度特征 |
| CFR 主网络入口 | `nets/yolo_tsaff.py` 中的 `YoloBody` | 串联双分支、TSAFF、PAFPN 和检测头 |
| 三尺度特征融合 | `nets/yolo_tsaff.py` 中的 `YOLOPAFPN2TSAFF` | 在 P3、P4、P5 三个尺度调用 TSAFF |
| TSAFF 融合模块 | `nets/tsaff_modified.py` 中的 `TSAFF` | 完成跨模态特征对齐、迭代细化与融合 |
| SFM | `nets/tsaff_modified.py` 中的 `SFM` | 压缩并统一 CT/PET 特征表示 |
| IFRM | `nets/tsaff_modified.py` 中的 `IFRM` | 迭代执行跨模态特征细化 |
| QVGSIM | `nets/tsaff_modified.py` 中的 `QVGSIM` | 建立查询引导的跨模态交互关系 |
| FUSM | `nets/tsaff_modified.py` 中的 `FUSM` | 对直接特征与细化特征进行选择性融合 |
| 多尺度特征金字塔 | `nets/yolo_tsaff.py` 的 P3/P4/P5 路径 | 聚合不同空间分辨率下的病灶信息 |
| YOLOX 检测头 | `nets/yolo.py` 中的 `YOLOXHead` | 输出类别、目标置信度和检测框 |
| 损失函数 | `nets/yolo_training.py` 中的 `YOLOLoss` | 计算分类、置信度与边界框回归损失 |
| 推理与后处理 | `yolo.py`、`utils/utils_bbox.py` | 解码预测、非极大值抑制并输出结果 |

## 3. TSAFF 内部过程

图2中的 TSAFF 可按以下顺序理解：

1. **特征标准化与压缩**：SFM 将两个模态的输入特征转换到统一的通道和空间表达。
2. **位置编码**：为序列化后的 CT/PET 特征加入可学习位置嵌入，使交互过程保留空间位置信息。
3. **跨模态交互**：QVGSIM 在两个模态之间建立查询引导的特征联系。
4. **迭代细化**：IFRM 复用 QVGSIM 参数进行多轮更新，逐步增强互补信息并抑制模态噪声。
5. **加权组合**：使用可学习系数组合原始特征、直接交互特征和迭代细化特征。
6. **选择性融合**：FUSM 分别处理两个模态分支，随后得到最终的融合特征。

对应实现集中在 `nets/tsaff_modified.py`。`nets/yolo_tsaff.py` 在三个特征尺度上分别创建 TSAFF，并将融合后的特征送入特征金字塔。

## 4. 前向传播的数据关系

主模型的调用方式为：

```python
from nets.yolo_tsaff import YoloBody

model = YoloBody(num_classes=1, phi="l", tsaff_loops=2)
outputs = model(ct_image, pet_image)
```

其中：

- `ct_image` 与 `pet_image` 应为空间对应的四维张量，形状为 `[batch, channels, height, width]`；
- `phi` 控制 YOLOX 主干的深度和宽度；
- `tsaff_loops` 控制 IFRM 的迭代细化次数；
- 输出由三个尺度的 YOLOX 检测结果组成。

## 5. 训练、测试与图2的关系

- `train_tsaff.py` 实例化图2所示完整网络，并通过 `utils/dataloader.py` 载入配对数据。
- `nets/yolo_training.py` 负责检测损失、学习率调度及指数滑动平均。
- `test_tsaff.py` 加载训练权重并在验证或测试数据上运行完整前向传播。
- `predict.py` 和根目录下的 `yolo.py` 提供推理接口。
- `get_map.py` 使用模型预测结果计算检测指标。

## 6. 文件命名说明

公开目录使用 `CFR_main_experiment_code`，以与论文方法名称一致。部分内部文件仍保留 `tsaff` 名称，因为 TSAFF 是 CFR 框架的核心融合模块，而且训练脚本与 Python 导入语句依赖这些名称。直接重命名这些内部文件会破坏原实验代码的可复现性。

