# Ablation Summary

| Experiment | Description | Best Epoch | Mean DSC | Median DSC |
| --- | --- | --- | --- | --- |
| Baseline | Standard U-Net with Dice loss and Adam. | 19 | 0.7992 | 0.9018 |
| Baseline + BatchNorm | Enable batch normalization in convolution blocks. | 8 | 0.8882 | 0.8960 |
| Baseline + Bilinear | Use bilinear upsampling in the decoder. | 17 | 0.8027 | 0.9074 |
| Baseline + Dropout | Add dropout to deeper encoder/decoder blocks. | 17 | 0.7973 | 0.9025 |
| Baseline + BCE+Dice | Replace Dice loss with combined BCE and Dice loss. | 19 | 0.9052 | 0.9216 |
| Baseline + AdamW | Replace Adam with AdamW and add mild weight decay. | 19 | 0.8038 | 0.9121 |
| Baseline + Cosine Scheduler | Add cosine annealing learning-rate scheduling. | 17 | 0.7914 | 0.8872 |
| Baseline + Stronger Aug | Increase scale and rotation augmentation strength. | 5 | 0.7847 | 0.8792 |
| All Improvements | Combine batch norm, bilinear upsampling, dropout, BCE+Dice, AdamW, cosine scheduling, and stronger augmentation. | 19 | 0.8098 | 0.9141 |
