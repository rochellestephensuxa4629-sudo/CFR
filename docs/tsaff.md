# Two-Stage Adaptive Feature Fusion

TSAFF is the cross-modal fusion component of CFR-Net. It receives paired CT and PET feature maps at the same pyramid level and produces a fused representation for the detection neck.

The implementation is located in `nets/tsaff_modified.py` and contains the following components:

- `SFM` compresses spatial features and produces compact modality-specific tokens.
- `QVGSIM` performs Q-V-gated semantic interaction between CT and PET features.
- `IFRM` applies the interaction module iteratively with shared parameters.
- `FUSM` suppresses uncertain spatial responses and refines lesion boundaries.
- `TSAFF` combines semantic alignment and localization refinement into one fusion block.

Three TSAFF blocks are instantiated in `nets/yolo_tsaff.py` for the P3, P4 and P5 feature levels. The fused outputs are processed by the feature pyramid and YOLOX detection head.

```python
from nets.yolo_tsaff import YoloBody

model = YoloBody(num_classes=1, phi="l", tsaff_loops=2)
outputs = model(ct_image, pet_image)
```

The `tsaff_loops` argument controls the number of iterative refinement steps. CT and PET inputs use the tensor layout `[batch, channels, height, width]` and must be spatially paired before they are passed to the model.

