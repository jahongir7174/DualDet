[DualDet](https://arxiv.org/abs/2107.00420) implementation using PyTorch

### Install

* `pip install mmcv-full`
* `pip install mmdet`

### Train

* `bash ./main.sh ./nets/exp01.py $ --train` for training, `$` is number of GPUs

### Test

* `bash ./main.sh ./nets/exp01.py $ --test` for testing, `$` is number of GPUs

### Results

|   Detector   | Backbone | Neck | LR Schedule | Box mAP | Config |
|:------------:|:--------:|:----:|:-----------:|--------:|-------:|
| Faster R-CNN |  Swin-T  | FPN  |     1x      |       - |  exp01 |
|   DualDet    |  Swin-T  | FPN  |     1x      |       - |  exp02 |
|   DualDet    |  Swin-T  | FPN  |     1x      |       - |  exp03 |

### TODO

* [x] [exp01](./nets/exp01.py), default [Faster R-CNN](https://arxiv.org/abs/1506.01497)
* [x] [exp02](./nets/exp02.py), added [DualDet](https://arxiv.org/abs/2107.00420)
* [x] [exp03](./nets/exp03.py), added [MOSAIC](https://arxiv.org/abs/2004.10934)
  |  [MixUp](https://arxiv.org/abs/1710.09412)

### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/open-mmlab/mmdetection