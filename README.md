# CFR Main Experiment Code

本目录是论文 CFR（Cross-Modal Feature Alignment and Localization Refinement）主模型的精简实验代码。代码结构与论文图2对应，不包含五折交叉验证、消融实验、对比模型实验、Bootstrap 分析和论文绘图工具。

图2各组成部分与源代码的对应关系见 [`FIGURE_2_CODE_GUIDE.md`](FIGURE_2_CODE_GUIDE.md)。

## 目录

- `nets/`：CFR 主模型、双分支特征提取网络及 TSAFF 跨模态融合模块。
- `utils/`：数据加载、训练、检测框处理和指标计算工具。
- `train_tsaff.py`：CFR 主模型训练入口（保留原文件名以确保代码兼容）。
- `test_tsaff.py`：CFR 主模型测试入口（保留原文件名以确保代码兼容）。
- `predict.py`：单张或批量预测入口。
- `yolo.py`：模型加载与推理封装。
- `get_map.py`：检测性能评估。
- `voc_annotation.py`、`to_VOC.py`：数据集格式准备工具。
- `requirements.txt`：原实验环境依赖。

## 使用前准备

1. 安装 `requirements.txt` 中的依赖。
2. 将私有 PET/CT 数据放在代码仓库之外。
3. 根据本机环境修改训练、测试脚本中的数据集路径和权重路径。
4. 不要上传患者影像、标注原始文件或包含隐私信息的结果。

## 快速运行

```bash
python train_tsaff.py
python test_tsaff.py
python predict.py
python get_map.py
```

上传前请补充作者信息、论文引用方式和合适的软件许可证。


