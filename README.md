[DualDet](https://arxiv.org/abs/2107.00420) implementation using PyTorch

### Install

* `pip install mmcv-full`
* `pip install mmdet`

### Train

* `bash ./main.sh ./nets/exp01.py $ --train` for training, `$` is number of GPUs

### Test

* `bash ./main.sh ./nets/exp01.py $ --test` for testing, `$` is number of GPUs

### Results

|   Detector   | Backbone | Neck | LR Schedule | Box mAP |                   Config |                                                                            Download |
|:------------:|:--------:|:----:|:-----------:|--------:|-------------------------:|------------------------------------------------------------------------------------:|
| Faster R-CNN |  Swin-T  | FPN  |     1x      |   42.86 | [exp01](./nets/exp01.py) | [model](https://github.com/jahongir7174/DualDet/releases/download/v0.0.1/exp01.pth) |
|   DualDet    |  Swin-T  | FPN  |     1x      |   47.06 | [exp02](./nets/exp02.py) | [model](https://github.com/jahongir7174/DualDet/releases/download/v0.0.1/exp02.pth) |

### TODO

* [x] [exp01](./nets/exp01.py), default [Faster R-CNN](https://arxiv.org/abs/1506.01497)
* [x] [exp02](./nets/exp02.py), added [DualDet](https://arxiv.org/abs/2107.00420)

### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/open-mmlab/mmdetection