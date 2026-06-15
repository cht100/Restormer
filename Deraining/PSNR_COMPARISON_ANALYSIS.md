# Restormer Deraining PSNR 计算链路对比审阅

本文只分析 PSNR。重点比较两条链路：

- Restormer 训练时验证日志里的 PSNR：`basicsr/train.py` 调用 `ImageCleanModel.validation()`。
- 自定义脚本里的 PSNR：`Deraining/test_restormer_metrics.py`。

结论先行：在“模型权重相同、数据集图像相同”的前提下，两边 PSNR 仍可能不同，主要原因不是模型结构，而是 **评测链路不一致**：训练验证默认在 `uint8` 图像上、转到 **YCbCr 的 Y 通道** 后计算；自定义脚本默认用 `pyiqa` 的 `psnr` 在 **RGB tensor [0,1]** 上计算。其次，训练验证使用 BasicSR dataloader 的配对规则和训练配置中的 val 路径，而自定义脚本使用官方 `test.py` 风格的固定五个 Rain 数据集路径。若实际运行的数据集目录不一致，PSNR 没有可比性。

## 1. Restormer 训练验证 PSNR 链路

### 1.1 训练入口与验证触发

入口是 `basicsr/train.py`。

- `parse_options()` 读取 YAML，并设置分布式、随机种子：`basicsr/train.py:25-59`。
- `create_train_val_dataloader()` 创建训练和验证 dataloader：`basicsr/train.py:83-129`。
- 训练主循环中，每 `val_freq` 触发一次验证：`basicsr/train.py:292-299`。
- 验证时传入：
  - `rgb2bgr = opt['val'].get('rgb2bgr', True)`：`basicsr/train.py:295`
  - `use_image = opt['val'].get('use_image', True)`：`basicsr/train.py:297`
  - `model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'], rgb2bgr, use_image)`：`basicsr/train.py:298-299`

当前 Deraining 配置为：

- 验证集 GT：`/data/users/user3/chen/16-baselines-restoration/datasets/RESIDE-6K/test/GT/`，见 `Deraining/Options/Deraining_Restormer.yml:46`
- 验证集 LQ：`/data/users/user3/chen/16-baselines-restoration/datasets/RESIDE-6K/test/hazy`，见 `Deraining/Options/Deraining_Restormer.yml:47`
- `val.window_size: 8`：`Deraining/Options/Deraining_Restormer.yml:106`
- `val.rgb2bgr: true`：`Deraining/Options/Deraining_Restormer.yml:109`
- `val.use_image: true`：`Deraining/Options/Deraining_Restormer.yml:110`
- PSNR 配置：
  - `type: calculate_psnr`
  - `crop_border: 0`
  - `test_y_channel: true`
  见 `Deraining/Options/Deraining_Restormer.yml:113-117`。

这意味着训练日志里的 PSNR 不是 RGB 全图 PSNR，而是 **Y 通道 PSNR**。

### 1.2 验证 dataloader 与数据读取

Deraining 配置使用 `Dataset_PairedImage`：`Deraining/Options/Deraining_Restormer.yml:43-47`。

创建逻辑：

- `create_dataset()` 根据 `type` 找到 `Dataset_PairedImage`：`basicsr/data/__init__.py:29-53`
- 验证 dataloader 固定 `batch_size=1, shuffle=False, num_workers=0`：`basicsr/data/__init__.py:99-101`

配对逻辑在 `paired_paths_from_folder()`：

- 扫描 LQ 和 GT 文件：`basicsr/data/data_util.py:232-233`
- 要求数量一致：`basicsr/data/data_util.py:234-236`
- 以 GT 文件名的 basename 为基准，用 `filename_tmpl` 构造输入文件名：`basicsr/data/data_util.py:238-246`
- 最终保存 `lq_path` 和 `gt_path`：`basicsr/data/data_util.py:247-250`

注意：`scandir()` 直接用 `os.scandir()`，没有显式排序：`basicsr/utils/misc.py:53-93`。不过该配对函数会以 GT 文件名构造输入文件名并检查是否存在，因此只要 LQ 与 GT 同名，配对本身通常是可靠的。遍历顺序会影响日志中单张图片处理顺序，但平均 PSNR 不受顺序影响。

图像读取与张量转换：

- `imfrombytes(..., float32=True)` 将 OpenCV 解码的 BGR 图变成 `float32 [0,1]`：`basicsr/utils/img_util.py:101-125`
- `Dataset_PairedImage.__getitem__()` 中 GT/LQ 都这样读取：`basicsr/data/paired_image_dataset.py:85-99`
- 验证阶段不做 crop/augmentation；只有训练阶段才 padding、random crop、augmentation：`basicsr/data/paired_image_dataset.py:101-114`
- `img2tensor(..., bgr2rgb=True)` 将 BGR HWC 变成 RGB CHW tensor：`basicsr/data/paired_image_dataset.py:115-118`，具体实现见 `basicsr/utils/img_util.py:9-33`

因此，训练验证输入网络的是 **RGB tensor，范围 [0,1]**。

### 1.3 验证推理逻辑

验证实现位于 `basicsr/models/image_restoration_model.py`。

- `validation()` 在单机或 rank 0 上调用 `nondist_validation()`：`basicsr/models/base_model.py:37-52` 与 `basicsr/models/image_restoration_model.py:207-214`
- 读取 `window_size`：`basicsr/models/image_restoration_model.py:224`
- 当前配置 `window_size=8`，所以使用 `pad_test()`：`basicsr/models/image_restoration_model.py:226-229`

`pad_test()` 做的事：

- 如果高/宽不是 8 的倍数，则在右侧、下侧 reflect padding：`basicsr/models/image_restoration_model.py:175-184`
- 推理后裁回原始尺寸：`basicsr/models/image_restoration_model.py:185-186`

`nonpad_test()` 做的事：

- 如果存在 EMA 网络，使用 `net_g_ema`：`basicsr/models/image_restoration_model.py:191-197`
- 否则使用 `net_g.eval()` 推理，并在结束后切回 train：`basicsr/models/image_restoration_model.py:198-205`

当前 Deraining 配置没有 `ema_decay`，所以默认使用 `net_g`，不是 EMA。

### 1.4 输出转图与 PSNR 计算

验证循环中：

- `visuals = self.get_current_visuals()`：`basicsr/models/image_restoration_model.py:239`
- 输出转图：`sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)`：`basicsr/models/image_restoration_model.py:240`
- GT 转图：`gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)`：`basicsr/models/image_restoration_model.py:241-242`

当前 `rgb2bgr=true`，所以 `tensor2img()` 会：

- clamp 到 `[0,1]`：`basicsr/utils/img_util.py:67`
- RGB 转 BGR：`basicsr/utils/img_util.py:84-85`
- 乘 255 并 round：`basicsr/utils/img_util.py:91-94`
- 输出 `np.uint8` BGR 图。

因为 `use_image=true`，PSNR 不是直接在 float tensor 上算，而是在上述 `uint8` 图像上算：

- `use_image` 分支调用 `metric_module.calculate_psnr(sr_img, gt_img, **opt_)`：`basicsr/models/image_restoration_model.py:273-280`

`calculate_psnr()` 位于 `basicsr/metrics/psnr_ssim.py`：

- 检查 shape：`basicsr/metrics/psnr_ssim.py:31-36`
- 转成 HWC：`basicsr/metrics/psnr_ssim.py:46-47`
- 转 `float64`：`basicsr/metrics/psnr_ssim.py:48-49`
- `crop_border=0`，不裁边：`basicsr/metrics/psnr_ssim.py:51-53`
- `test_y_channel=true` 时转 Y 通道：`basicsr/metrics/psnr_ssim.py:55-57`
- MSE 后用 `20 * log10(max_value / sqrt(mse))`：`basicsr/metrics/psnr_ssim.py:59-63`

Y 通道转换：

- `to_y_channel()` 先将 `[0,255]` 除以 255 成 `[0,1]`：`basicsr/metrics/metric_util.py:43`
- 如果是 3 通道，则调用 `bgr2ycbcr(..., y_only=True)`：`basicsr/metrics/metric_util.py:44-46`
- `bgr2ycbcr()` 的 Y 公式为 `24.966*B + 128.553*G + 65.481*R + 16`：`basicsr/utils/matlab_functions.py:207-238`

所以训练验证 PSNR 的准确表达是：

```text
输入 LQ/GT: cv2 解码 BGR uint8 -> float32 [0,1] -> RGB tensor
推理: reflect pad 到 8 倍数 -> Restormer -> 裁回原图
输出/GT: RGB tensor -> clamp [0,1] -> BGR uint8 round
PSNR: BGR uint8 -> Matlab 风格 Y 通道 -> MSE -> PSNR，crop_border=0
聚合: 每张图 PSNR 简单平均
```

## 2. 自定义 `test_restormer_metrics.py` PSNR 链路

### 2.1 入口与模型配置

自定义脚本位于 `Deraining/test_restormer_metrics.py`。

参数：

- `--input_dir` 默认 `/home/quyu/datasets/Rain13K/`：`Deraining/test_restormer_metrics.py:456`
- `--weights` 默认 `/home/quyu/16-baselines-restoration/models/Restormer/Derain/net_g_300000.pth`：`Deraining/test_restormer_metrics.py:463`
- `--crop_border` 默认 0：`Deraining/test_restormer_metrics.py:467`

模型 YAML 写死为：

- `yaml_file = "Deraining/Options/Deraining_Restormer_test.yml"`：`Deraining/test_restormer_metrics.py:526`

但当前仓库实际只有：

- `Deraining/Options/Deraining_Restormer.yml`

没有 `Deraining_Restormer_test.yml`。如果运行目录下没有额外文件，脚本会在 `yaml.load(open(yaml_file...))` 处失败。这是一个真实代码风险。

加载权重：

- 如果 checkpoint 是 dict 且有 `params`，使用 `checkpoint["params"]`，否则直接使用 checkpoint：`Deraining/test_restormer_metrics.py:538-540`
- 如果有 CUDA，则包 `nn.DataParallel`：`Deraining/test_restormer_metrics.py:543-547`

### 2.2 数据集枚举与 GT 查找

自定义脚本固定评测五个数据集：

```python
datasets = ["Rain100L", "Rain100H", "Test100", "Test1200", "Test2800"]
```

位置：`Deraining/test_restormer_metrics.py:557-558`。

每个数据集输入目录为：

- `{args.input_dir}/test/{dataset}/input`：`Deraining/test_restormer_metrics.py:567`

文件收集：

- 使用 `natsorted(glob(...*.png) + glob(...*.jpg))`：`Deraining/test_restormer_metrics.py:568`

GT 目录查找：

- 在输入目录的上一级里找 `target/gt/GT/groundtruth/ground_truth/label/clean`：`Deraining/test_restormer_metrics.py:488-495`

单张 GT 文件查找：

- 优先同名文件：`Deraining/test_restormer_metrics.py:475-478`
- 再按相同 stem 试 `.png/.jpg/.jpeg/.bmp/.tif/.tiff`：`Deraining/test_restormer_metrics.py:480-485`

这和训练验证配置不同：训练配置当前验证的是 RESIDE-6K 的 `test/hazy` 与 `test/GT`，不是固定五个 Rain 数据集。即使权重相同，只要 `--input_dir` 没有指向同一套图像，PSNR 不能直接比较。

### 2.3 自定义脚本推理逻辑

读取输入：

- `utils.load_img(file_)` 使用 `cv2.imread()` 后 BGR 转 RGB：`Deraining/utils.py:80-81`
- `lq_rgb_uint8 = utils.load_img(file_)`：`Deraining/test_restormer_metrics.py:621`
- `img = np.float32(lq_rgb_uint8) / 255.0`：`Deraining/test_restormer_metrics.py:624`
- `permute(2,0,1)` 后送入模型：`Deraining/test_restormer_metrics.py:625-626`

这与训练验证的数据读取在数值上基本一致：两者最终都是 RGB tensor `[0,1]`。

padding：

- `factor=8`：`Deraining/test_restormer_metrics.py:557`
- 右侧/下侧 reflect pad：`Deraining/test_restormer_metrics.py:628-633`
- 推理后裁回原尺寸：`Deraining/test_restormer_metrics.py:640-642`

这与训练验证的 `pad_test(window_size=8)` 基本一致。

输出：

- clamp 到 `[0,1]`：`Deraining/test_restormer_metrics.py:642`
- 转 HWC RGB numpy：`Deraining/test_restormer_metrics.py:642`
- `img_as_ubyte(restored)` 转 RGB uint8：`Deraining/test_restormer_metrics.py:645`
- 保存时 `utils.save_img()` 会 RGB 转 BGR 后 `cv2.imwrite()`：`Deraining/utils.py:83-84`
- 指标计算前，脚本又把 RGB uint8 转成 BGR uint8：`Deraining/test_restormer_metrics.py:649`

### 2.4 自定义脚本 PSNR 计算

指标初始化：

- `pyiqa.create_metric("psnr", device=device)`：`Deraining/test_restormer_metrics.py:498-504`

单张图计算：

- `compute_one_output_metrics()` 中先将 BGR uint8 转 RGB tensor `[0,1]`：`Deraining/test_restormer_metrics.py:315-321`
- 对 `("LPIPS", "PSNR", "SSIM")` 调用 `fr_metrics[metric](sr_crop_rgb, gt_crop_rgb)`：`Deraining/test_restormer_metrics.py:322-325`

关键点：该脚本没有传入 Y 通道配置，也没有调用 Restormer BasicSR 的 `calculate_psnr()`。因此它的 PSNR 语义是 **pyiqa psnr 对 RGB tensor 的计算**，不是训练验证的 **BasicSR Y-channel PSNR**。

聚合方式：

- `add_metric()` 将每张图 PSNR 累加、计数：`Deraining/test_restormer_metrics.py:275-283`
- `compute_averages()` 做简单平均：`Deraining/test_restormer_metrics.py:286-298`

聚合方式与训练验证一样都是“先算每张图 PSNR，再平均”。差异主要在每张图 PSNR 的定义和数据路径。

## 3. 两条链路的关键差异清单

### 3.1 数据集路径差异：影响最大，必须先排除

训练配置当前验证：

```text
/data/users/user3/chen/16-baselines-restoration/datasets/RESIDE-6K/test/hazy
/data/users/user3/chen/16-baselines-restoration/datasets/RESIDE-6K/test/GT
```

见 `Deraining/Options/Deraining_Restormer.yml:46-47`。

自定义脚本默认评测：

```text
/home/quyu/datasets/Rain13K/test/Rain100L/input
/home/quyu/datasets/Rain13K/test/Rain100H/input
/home/quyu/datasets/Rain13K/test/Test100/input
/home/quyu/datasets/Rain13K/test/Test1200/input
/home/quyu/datasets/Rain13K/test/Test2800/input
```

见 `Deraining/test_restormer_metrics.py:456` 与 `Deraining/test_restormer_metrics.py:557-568`。

如果这两个不是同一批图像，PSNR 差异没有分析意义。

### 3.2 PSNR 颜色空间差异：训练是 Y 通道，自定义脚本是 RGB

训练验证：

- `test_y_channel: true`：`Deraining/Options/Deraining_Restormer.yml:117`
- 使用 `to_y_channel()`：`basicsr/metrics/psnr_ssim.py:55-57`
- 使用 BGR 版 Matlab YCbCr：`basicsr/metrics/metric_util.py:43-47` 与 `basicsr/utils/matlab_functions.py:207-238`

自定义脚本：

- 计算前转为 RGB tensor `[0,1]`：`Deraining/test_restormer_metrics.py:315-321`
- 调用 `pyiqa psnr`：`Deraining/test_restormer_metrics.py:322-325`
- 没有 Y channel 参数。

这会明显影响 PSNR。Y 通道只评价亮度误差，RGB PSNR 对三个通道整体误差求均值。对于颜色偏差、白平衡偏差、色彩残留明显的输出，两者可能差很多。

### 3.3 uint8 量化时机差异

训练验证：

- 输出和 GT 先经 `tensor2img()` 转为 uint8：`basicsr/models/image_restoration_model.py:239-242`
- `tensor2img()` 会 clamp、乘 255、round、astype uint8：`basicsr/utils/img_util.py:67-94`
- 然后在 uint8 图上算 PSNR。

自定义脚本：

- 输出先 `img_as_ubyte(restored)` 转 uint8：`Deraining/test_restormer_metrics.py:645`
- GT 是 `utils.load_img()` 读出的 uint8：`Deraining/test_restormer_metrics.py:656`
- 再转 RGB tensor `[0,1]` 给 pyiqa：`Deraining/test_restormer_metrics.py:315-321`

两边都发生了 uint8 化，但使用的转换函数不同：BasicSR 是显式 `.round()`；自定义脚本是 `skimage.img_as_ubyte()`。通常差异很小，但在边界像素和四舍五入规则上可能造成极小 PSNR 差别。

如果想完全对齐训练日志，建议直接复用 `basicsr.metrics.calculate_psnr()` 和 `tensor2img()` 的输出，而不是混用 `img_as_ubyte + pyiqa`。

### 3.4 padding 推理基本一致

训练验证：

- `window_size=8`：`Deraining/Options/Deraining_Restormer.yml:106`
- `F.pad(..., 'reflect')`：`basicsr/models/image_restoration_model.py:175-184`
- 输出裁回原尺寸：`basicsr/models/image_restoration_model.py:185-186`

自定义脚本：

- `factor=8`：`Deraining/test_restormer_metrics.py:557`
- `F.pad(..., "reflect")`：`Deraining/test_restormer_metrics.py:628-633`
- 输出裁回原尺寸：`Deraining/test_restormer_metrics.py:640-642`

这部分基本对齐。只要模型结构、权重、输入图完全一样，padding 本身通常不是 PSNR 差异来源。

### 3.5 模型权重 key 和 EMA 差异

训练验证中的模型：

- `ImageCleanModel` 加载 `pretrain_network_g` 时默认 `param_key='params'`：`basicsr/models/image_restoration_model.py:69-73`
- 训练验证若存在 `net_g_ema` 则用 EMA，否则用 `net_g`：`basicsr/models/image_restoration_model.py:188-205`
- 当前配置没有 `ema_decay`，所以不是 EMA。

自定义脚本：

- 优先使用 checkpoint 的 `params`：`Deraining/test_restormer_metrics.py:538-540`
- 不读取 `params_ema`。

当前 Deraining 配置下二者基本一致。但如果未来权重文件同时含 `params` 和 `params_ema`，训练验证是否使用 EMA 取决于训练配置；自定义脚本固定偏向 `params`，可能不一致。

### 3.6 文件配对和命名差异

训练验证：

- 以 GT basename 构造 LQ 文件名，并检查存在：`basicsr/data/data_util.py:238-246`

自定义脚本：

- 遍历 input 文件，再查找同名或同 stem GT：`Deraining/test_restormer_metrics.py:471-485`

只要 LQ/GT 完全同名或同 stem，二者都能正确配对。若命名规则不同，例如 LQ 有后缀、GT 无后缀，训练配置可以通过 `filename_tmpl` 对齐；自定义脚本目前不支持复杂模板，可能找错或找不到 GT。

### 3.7 crop_border 当前一致

训练验证：

- `crop_border: 0`：`Deraining/Options/Deraining_Restormer.yml:116`

自定义脚本：

- `--crop_border` 默认 0：`Deraining/test_restormer_metrics.py:467`
- `crop_pair()` 中只有大于 0 才裁剪：`Deraining/test_restormer_metrics.py:217-223`

默认情况下一致。

## 4. 在同权重同数据下，哪些因素会改变 PSNR 大小

按影响优先级排序：

1. **评测图像是否真的是同一批、同一 GT 配对**  
   训练配置是 RESIDE-6K val；自定义脚本默认是 Rain13K 五个 test 子集。不同数据集会导致 PSNR 完全不可比。

2. **Y 通道 PSNR vs RGB PSNR**  
   训练日志是 Y 通道；自定义脚本是 RGB tensor。亮度误差和 RGB 全通道误差不是同一个指标。

3. **是否在 uint8 图上算**  
   训练日志明确是 uint8 后算，因为 `use_image=true`。如果改成 tensor float 计算，PSNR 会不同。自定义脚本虽然也先生成 uint8 输出，但最终通过 pyiqa 的 RGB tensor 计算。

4. **输出 clamp/round 的实现细节**  
   `tensor2img()` 与 `img_as_ubyte()` 的量化细节可能带来很小差异。

5. **padding 是否一致**  
   当前两边都是右/下 reflect pad 到 8 倍数并裁回，基本一致。若一边不 pad 或使用 zero/replicate pad，边缘输出会变，PSNR 会变。

6. **模型是否用 EMA 权重**  
   当前配置不启用 EMA；若启用，训练验证可能用 `net_g_ema`，自定义脚本仍用 `params`，PSNR 会变。

7. **数据读取颜色顺序**  
   当前两边输入网络前都变成 RGB `[0,1]`，这点基本一致。训练 PSNR 中的 Y 转换基于 BGR uint8，是因为 `rgb2bgr=true` 后再 `bgr2ycbcr()`，逻辑上是自洽的。

## 5. 如何让自定义脚本对齐训练日志 PSNR

若目标是复现 `basicsr/train.py` 训练时日志里的 PSNR，建议把自定义脚本的 PSNR 改成以下逻辑：

```python
from basicsr.metrics.psnr_ssim import calculate_psnr

psnr = calculate_psnr(
    sr_img_bgr_uint8,
    gt_img_bgr_uint8,
    crop_border=0,
    input_order="HWC",
    test_y_channel=True,
)
```

并确保：

- 使用和训练配置相同的验证路径。
- 使用同一套 LQ/GT 配对规则。
- 输出图是 clamp 后 uint8 round 的图；最好直接使用 BasicSR `tensor2img(..., rgb2bgr=True)`。
- `window_size/factor=8`，reflect pad 到 8 倍数后裁回。
- 权重 key 与训练验证一致：当前为 `params`，不要误用 `params_ema`。

如果仍想保留 pyiqa 的 PSNR，可以同时记录两个名字，例如：

- `psnr_y_basicsr`: 对齐训练日志。
- `psnr_rgb_pyiqa`: 自定义 RGB 指标。

这样不会把两个不同定义的 PSNR 混在一起。

## 6. 最重要的审阅结论

当前代码下，训练日志 PSNR 与自定义脚本 PSNR 不应直接比较。

训练日志 PSNR 的定义是：

```text
Restormer val set output -> uint8 BGR -> Matlab-style Y channel -> PSNR -> dataset mean
```

自定义脚本 PSNR 的定义是：

```text
official test.py output -> uint8 RGB/BGR 转换 -> RGB tensor [0,1] -> pyiqa psnr -> dataset mean
```

如果“模型权重相同、数据集一样”但 PSNR 仍不同，最应优先检查的是：

1. 自定义脚本是否真的评测训练配置里的 RESIDE-6K val，而不是默认 Rain13K 五个子集。
2. 自定义脚本是否按 Y 通道计算，而不是 RGB。
3. 自定义脚本是否复用了 BasicSR 的 `calculate_psnr()`。
4. `Deraining_Restormer_test.yml` 是否存在；当前仓库中不存在这个文件。

