# HiSCo-Net modules

HiSCo-Net comprises RCSGF, RGDU, and BiRSC. Detailed module implementations are not included in this pre-publication release. Source code and model inference support are planned for release after paper acceptance.

Model YAMLs and full-object checkpoints are published separately in `models/` and `weights/`. They expose architecture-level information but do not replace the module definitions required for inference. The public `evaluate.py` evaluates existing COCO prediction JSONs without importing these modules.
