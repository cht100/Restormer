# Restormer SR 训练

当前仓库里的 Restormer 官方代码没有独立 SR 上采样结构，`ImageCleanModel` 直接执行 `net_g(lq)`，输出尺寸和输入尺寸一致。因此这里不改模型结构，直接复用 RDDM-SR 已经生成好的同尺寸 SR 数据对：

```text
input = bicubic(LR, x4)
gt    = HR
```

也就是说，Restormer-SR 不需要重新生成数据。只要 RDDM-SR 的数据对已经存在，改好 `SR/Options/SR_Restormer.yml` 里的四个路径即可：

```yaml
datasets:
  train:
    dataroot_lq: /data1/jiangtaoren/16-baselines-restoration/methods/RDDM/experiments/2_Image_Restoration_sr/data/DIV2K_x4_sr/train/input
    dataroot_gt: /data1/jiangtaoren/16-baselines-restoration/methods/RDDM/experiments/2_Image_Restoration_sr/data/DIV2K_x4_sr/train/gt
  val:
    dataroot_lq: /data1/jiangtaoren/16-baselines-restoration/methods/RDDM/experiments/2_Image_Restoration_sr/data/DIV2K_x4_sr/test/input
    dataroot_gt: /data1/jiangtaoren/16-baselines-restoration/methods/RDDM/experiments/2_Image_Restoration_sr/data/DIV2K_x4_sr/test/gt
```

注意：因为输入已经是放大到 HR 尺寸后的图，所以 YAML 中 `scale: 1`、`gt_size: 256` 是和 Restormer 当前官方训练代码匹配的；这里不要改成 `scale: 4`。

训练过程中的 validation 会通过 `datasets.val.resize_to: 256` 把验证 input/GT 都缩到 256x256，只用于训练中监控，避免 DIV2K 大图整图验证导致显存溢出。最终论文表格数值仍然用统一 SR 测试脚本计算。

## 启动训练

```bash
cd /data1/jiangtaoren/16-baselines-restoration/methods/Restormer

CUDA_VISIBLE_DEVICES=3 python SR/train.py \
  -opt SR/Options/SR_Restormer.yml
```

训练权重会按 Restormer/BasicSR 默认逻辑保存到：

```text
methods/Restormer/experiments/SR_Restormer_x4_bicubic_input/models/
```

常用权重是 `net_g_latest.pth` 或对应迭代数的 `net_g_*.pth`。
