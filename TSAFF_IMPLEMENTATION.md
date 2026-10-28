# TSAFF implementation

New files:

- `nets/tsaff_modified.py`: TSAFF, SFM, IFRM, QVGSIM, and FUSM.
- `nets/yolo_tsaff.py`: YOLOX entry point using the new TSAFF module.
- `train_tsaff.py`: training entry point importing the new YOLOX body.
- `check_tsaff.py`: CPU/GPU smoke test for a CT/PET forward pass.

The original files are not overwritten. The new model keeps the same external
forward signature:

```python
from nets.yolo_tsaff import YoloBody

model = YoloBody(num_classes=1, phi="l", tsaff_loops=2)
outputs = model(ct_image, pet_image)
```

Run the smoke test before starting training:

```bash
python check_tsaff.py --phi l --loops 2 --size 640
```

The TSAFF setting follows the architecture diagrams:

1. SFM compresses CT and PET features.
2. Learnable positional embeddings are added.
3. IFRM reuses the same QVGSIM parameters for `n` iterations.
4. The four coefficients `zeta`, `epsilon`, `phi`, and `eta` combine direct
   and refined features.
5. Two branch-specific FUSM operations are followed by fusion and one final
   FUSM operation.
