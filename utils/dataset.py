import math
import random
from copy import deepcopy
from os.path import basename

import cv2
import mmcv
import numpy
from PIL import Image, ImageOps, ImageEnhance
from mmdet.datasets import DATASETS, CocoDataset
from mmdet.datasets.pipelines.compose import Compose, PIPELINES

max_value = 10.


def resample():
    return random.choice((Image.BILINEAR, Image.BICUBIC))


def rotate(image, magnitude):
    magnitude = (magnitude / max_value) * 45.0

    if random.random() > 0.5:
        magnitude *= -1

    return image.rotate(magnitude, resample=resample())


def shear_x(image, magnitude):
    magnitude = (magnitude / max_value) * 0.30

    if random.random() > 0.5:
        magnitude *= -1

    return image.transform(image.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0), resample=resample())


def shear_y(image, magnitude):
    magnitude = (magnitude / max_value) * 0.30

    if random.random() > 0.5:
        magnitude *= -1

    return image.transform(image.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0), resample=resample())


def translate_x(image, magnitude):
    magnitude = (magnitude / max_value) * 0.45

    if random.random() > 0.5:
        magnitude *= -1

    pixels = magnitude * image.size[0]
    return image.transform(image.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=resample())


def translate_y(image, magnitude):
    magnitude = (magnitude / max_value) * 0.45

    if random.random() > 0.5:
        magnitude *= -1

    pixels = magnitude * image.size[1]
    return image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=resample())


def equalize(image, _):
    return ImageOps.equalize(image)


def invert(image, _):
    return ImageOps.invert(image)


def identity(image, _):
    return image


def normalize(image, _):
    return ImageOps.autocontrast(image)


def brightness(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Brightness(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Brightness(image).enhance(magnitude)


def color(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Color(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Color(image).enhance(magnitude)


def contrast(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Contrast(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Contrast(image).enhance(magnitude)


def sharpness(image, magnitude):
    if random.random() > 0.5:
        magnitude = (magnitude / max_value) * 1.8 + 0.1
        return ImageEnhance.Sharpness(image).enhance(magnitude)
    else:
        magnitude = (magnitude / max_value) * 0.9

        if random.random() > 0.5:
            magnitude *= -1

        return ImageEnhance.Sharpness(image).enhance(magnitude)


def solar(image, magnitude):
    magnitude = int((magnitude / max_value) * 256)
    if random.random() > 0.5:
        return ImageOps.solarize(image, magnitude)
    else:
        return ImageOps.solarize(image, 256 - magnitude)


def poster(image, magnitude):
    magnitude = int((magnitude / max_value) * 4)
    if random.random() > 0.5:
        if magnitude >= 8:
            return image
        return ImageOps.posterize(image, magnitude)
    else:
        if random.random() > 0.5:
            magnitude = 4 - magnitude
        else:
            magnitude = 4 + magnitude

        if magnitude >= 8:
            return image
        return ImageOps.posterize(image, magnitude)


@PIPELINES.register_module()
class RandomMix:
    def __init__(self, mean=1, sigma=0.5, n=3):
        self.n = n
        self.mean = mean
        self.sigma = sigma
        self.transform = (equalize, identity, invert, normalize,
                          rotate, shear_x, shear_y, translate_x, translate_y,
                          brightness, color, contrast, sharpness, solar, poster)

    def __call__(self, results):
        image = results['img']
        image = mmcv.bgr2rgb(image)
        image = Image.fromarray(image)

        aug_image = image.copy()

        for transform in numpy.random.choice(self.transform, self.n):
            magnitude = numpy.random.normal(self.mean, self.sigma)
            magnitude = min(max_value, max(0., magnitude))
            aug_image = transform(aug_image, magnitude)

        alpha = random.random()
        image = Image.blend(image, aug_image, alpha if alpha > 0.4 else alpha / 2)
        image = mmcv.rgb2bgr(numpy.asarray(image))

        results['img'] = image
        return results


@PIPELINES.register_module()
class Weather:
    def __init__(self):
        import albumentations as album
        self.transform = album.OneOf([album.RandomRain(drop_width=1,
                                                       blur_value=5),
                                      album.RandomFog(fog_coef_lower=0.1,
                                                      fog_coef_upper=0.3),
                                      album.RandomShadow(num_shadows_lower=1,
                                                         num_shadows_upper=1),
                                      album.RandomSunFlare(angle_lower=0.5),
                                      album.RandomSnow(snow_point_lower=0.1,
                                                       snow_point_upper=0.3,
                                                       brightness_coeff=2.5)])

    def __call__(self, results):
        image = results['img']
        image = mmcv.bgr2rgb(image)
        results['img'] = self.transform(image=image)['image']
        return results


@PIPELINES.register_module()
class GridMask:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):
        if random.random() > self.p:
            return results

        image = results['img']
        shape = results['img_shape'][:2]

        h = int(shape[0] * 1.5)
        w = int(shape[1] * 1.5)
        d = numpy.random.randint(2, min(shape))

        st_h = numpy.random.randint(d)
        st_w = numpy.random.randint(d)
        mask = numpy.ones((h, w), numpy.float32)

        for i in range(h // d):
            s = d * i + st_h
            t = min(s + min(max(int(d / 2 + 0.5), 1), d - 1), h)
            mask[s:t, :] *= 0

        for i in range(w // d):
            s = d * i + st_w
            t = min(s + min(max(int(d / 2 + 0.5), 1), d - 1), w)
            mask[:, s:t] *= 0

        delta_h = (h - shape[0]) // 2
        delta_w = (w - shape[1]) // 2

        mask = mask[delta_h:delta_h + shape[0], delta_w:delta_w + shape[1]]

        mask = 1 - mask.astype(numpy.float32)
        mask = numpy.expand_dims(mask, 2).repeat(3, axis=2)

        results['img'] = (image * mask).astype('uint8')
        return results


@DATASETS.register_module()
class MyDataset(CocoDataset):
    CLASSES = ('class-0', 'class-1')


def resize(image, image_size):
    h, w = image.shape[:2]
    ratio = image_size / max(h, w)
    if ratio != 1:
        shape = (int(w * ratio), int(h * ratio))
        image = cv2.resize(image, shape, interpolation=random.choice((0, 1, 2, 3)))
    return image, image.shape[:2]


def random_hsv(image):
    # HSV color-space augmentation
    r = numpy.random.uniform(-1, 1, 3) * [0.015, 0.7, 0.4] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype('uint8')
    lut_sat = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_val = numpy.clip(x * r[2], 0, 255).astype('uint8')

    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def box_candidates(box1, box2):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    area = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.01) & (area < 20)


def random_perspective(image, boxes, size):
    # Center
    center = numpy.eye(3)
    center[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotation = numpy.eye(3)
    a = random.uniform(-10, 10)
    s = random.uniform(1 - 0.5, 1 + 0.5)
    rotation[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-10, 10) * math.pi / 180)  # x shear (deg)
    shear[1, 0] = math.tan(random.uniform(-10, 10) * math.pi / 180)  # y shear (deg)

    # Translation
    translation = numpy.eye(3)
    translation[0, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * size  # x translation (pixels)
    translation[1, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * size  # y translation (pixels)

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translation @ shear @ rotation @ perspective @ center
    if (matrix != numpy.eye(3)).any():
        image = cv2.warpAffine(image, matrix[:2], dsize=(size, size))  # affine

    if len(boxes):
        xy = numpy.ones((len(boxes) * 4, 3))
        # x1y1, x2y2, x1y2, x2y1
        xy[:, :2] = boxes[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(len(boxes) * 4, 2)
        xy = xy @ matrix.T  # transform
        xy = (xy[:, :2]).reshape(len(boxes), 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new_boxes = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, len(boxes)).T

        # clip
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, image.shape[1])
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, image.shape[0])

        # filter candidates
        candidates = box_candidates(boxes[:, 1:5].T * s, new_boxes.T)
        boxes = boxes[candidates]
        boxes[:, 1:5] = new_boxes[candidates]
    return image, boxes


def wh2xy(x, w, h, pad_w, pad_h):
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def mosaic(self, index, size=None):
    if size is None:
        size = numpy.random.choice(self.image_sizes)

    xc = int(random.uniform(size // 2, 2 * size - size // 2))
    yc = int(random.uniform(size // 2, 2 * size - size // 2))

    indexes4 = [index] + random.choices(range(self.num_samples), k=3)
    numpy.random.shuffle(indexes4)

    results4 = [deepcopy(self.dataset[index]) for index in indexes4]
    filename = results4[0]['filename']

    boxes4 = []
    shapes = [x['img_shape'][:2] for x in results4]
    image4 = numpy.full((2 * size, 2 * size, 3), 0, numpy.uint8)

    for i, (results, shape) in enumerate(zip(results4, shapes)):
        image, (h, w) = resize(results['img'], size)

        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, size * 2), min(size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        boxes = []
        label = numpy.array(results['ann_info']['labels'])

        for box in results['ann_info']['bboxes']:
            x_min, y_min, x_max, y_max = list(map(int, box))
            x_cen = (x_min + x_max) / (2 * shape[1])
            y_cen = (y_min + y_max) / (2 * shape[0])
            boxes.append([x_cen, y_cen, (x_max - x_min) / shape[1], (y_max - y_min) / shape[0]])

        if (len(label) == len(boxes)) and len(label) and len(boxes):
            boxes = numpy.array(boxes).astype('float32')
            boxes = (label.reshape(-1, 1), boxes)
            boxes = numpy.concatenate(boxes, axis=1)

            boxes[:, 1:] = wh2xy(boxes[:, 1:], w, h, pad_w, pad_h)

            boxes4.append(boxes)

    # concatenate & clip
    boxes4 = numpy.concatenate(boxes4, 0)
    for box4 in boxes4[:, 1:]:
        numpy.clip(a=box4, a_min=0, a_max=2 * size, out=box4)

    image4, boxes4 = random_perspective(image4, boxes4, size)

    label = []
    boxes = []
    for box4 in boxes4:
        label.append(box4[0])
        boxes.append(box4[1:5])
    if not len(boxes):
        return None
    # del copied results
    del results4
    if len(boxes) == len(label):
        random_hsv(image4)
        label = numpy.array(label, dtype=numpy.int64)
        boxes = numpy.array(boxes, dtype=numpy.float32)
        return dict(filename=filename, image=image4, label=label, boxes=boxes)
    else:
        return None


def mix_up(self, index1, index2):
    size = numpy.random.choice(self.image_sizes)

    data1 = mosaic(self, index1, size)
    data2 = mosaic(self, index2, size)
    alpha = numpy.random.beta(32.0, 32.0)

    if data1 is not None and data2 is not None:
        image1 = data1['image']
        label1 = data1['label']
        boxes1 = data1['boxes']

        image2 = data2['image']
        label2 = data2['label']
        boxes2 = data2['boxes']

        image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
        boxes = numpy.concatenate((boxes1, boxes2), 0)
        label = numpy.concatenate((label1, label2), 0)

        return dict(filename=data1['filename'], image=image, label=label, boxes=boxes)
    if data1 is None and data2 is not None:
        image = data2['image']
        label = data2['label']
        boxes = data2['boxes']

        return dict(filename=data2['filename'], image=image, label=label, boxes=boxes)
    if data1 is not None and data2 is None:
        image = data1['image']
        label = data1['label']
        boxes = data1['boxes']

        return dict(filename=data1['filename'], image=image, label=label, boxes=boxes)
    return None


def cut_mix(self, index1, index2):
    size = numpy.random.choice(self.image_sizes)

    data1 = mosaic(self, index1, size)
    data2 = mosaic(self, index2, size)

    if data1 is not None and data2 is not None:
        image1 = data1['image']
        image2 = data2['image']

        boxes1 = (data1['label'].reshape(-1, 1), data1['boxes'])
        boxes1 = numpy.concatenate(boxes1, axis=1)

        boxes2 = (data2['label'].reshape(-1, 1), data2['boxes'])
        boxes2 = numpy.concatenate(boxes2, axis=1)

        mix_image = image1.copy()
        size = min(image1.shape[0], image2.shape[0])
        x1, y1 = [int(random.uniform(size * 0.0, size * 0.45)) for _ in range(2)]
        x2, y2 = [int(random.uniform(size * 0.55, size * 1.0)) for _ in range(2)]

        mix_boxes = boxes2.copy()
        area = (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 4] - boxes2[:, 2])

        mix_boxes[:, [1, 3]] = mix_boxes[:, [1, 3]].clip(min=x1, max=x2)
        mix_boxes[:, [2, 4]] = mix_boxes[:, [2, 4]].clip(min=y1, max=y2)
        mix_boxes = mix_boxes.astype(numpy.int32)
        # cropped w, h, area
        w = mix_boxes[:, 3] - mix_boxes[:, 1]
        h = mix_boxes[:, 4] - mix_boxes[:, 2]
        area0 = w * h
        ar = numpy.maximum(w / (h + 1e-16), h / (w + 1e-16))

        mix_boxes = mix_boxes[numpy.where((w > 2) & (h > 2) & (area / (area0 + 1e-16) > 0.2) & (ar < 20))]
        mix_image[y1:y2, x1:x2] = (mix_image[y1:y2, x1:x2] + image2[y1:y2, x1:x2]) / 2

        mix_boxes = numpy.concatenate((boxes1, mix_boxes), axis=0)
        label = []
        boxes = []
        for mix_box in mix_boxes:
            label.append(mix_box[0])
            boxes.append(mix_box[1:5])

        if not len(boxes):
            return None

        if len(boxes) == len(label):
            label = numpy.array(label, dtype=numpy.int64)
            boxes = numpy.array(boxes, dtype=numpy.float32)

        return dict(filename=data1['filename'], image=mix_image, label=label, boxes=boxes)
    if data1 is None and data2 is not None:
        image = data2['image']
        label = data2['label']
        boxes = data2['boxes']

        return dict(filename=data2['filename'], image=image, label=label, boxes=boxes)
    if data1 is not None and data2 is None:
        image = data1['image']
        label = data1['label']
        boxes = data1['boxes']

        return dict(filename=data1['filename'], image=image, label=label, boxes=boxes)
    return None


def process(self, data):
    image = data['image']
    label = data['label']
    boxes = data['boxes']

    shape = image.shape

    results = dict()
    results['filename'] = data['filename']
    results['img_info'] = {'height': shape[0], 'width': shape[1]}
    results['ann_info'] = {'labels': label, 'bboxes': boxes}
    results['bbox_fields'] = []
    results['mask_fields'] = []
    results['ori_filename'] = basename(data['filename'])
    results['img'] = image
    results['img_fields'] = ['img']
    results['img_shape'] = shape
    results['ori_shape'] = shape
    results['pad_shape'] = shape

    results['scale_factor'] = numpy.array([1, 1, 1, 1], dtype=numpy.float32)
    return self.pipeline(results)


@DATASETS.register_module()
class MOSAICDataset:
    def __init__(self, dataset, image_sizes, pipeline, mix_p=0.0, cut_p=0.0):
        self.mix_p = mix_p
        self.cut_p = cut_p
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.pipeline = Compose(pipeline)
        if hasattr(self.dataset, 'flag'):
            self.flag = numpy.zeros(len(dataset), dtype=numpy.uint8)
        self.image_sizes = image_sizes
        self.num_samples = len(dataset)
        self.indices = range(len(dataset))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        while True:
            if self.cut_p:
                if random.random() > self.mix_p:
                    data = mix_up(self, index, random.choice(self.indices))
                else:
                    data = cut_mix(self, index, random.choice(self.indices))
            else:
                if random.random() > self.mix_p:
                    data = mosaic(self, index)
                else:
                    data = mix_up(self, index, random.choice(self.indices))

            if data is None:
                index = random.choice(self.indices)
                continue

            return process(self, data)


def build_dataset(cfg, default_args=None):
    if cfg['type'] == 'MOSAICDataset':
        import copy
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        return MOSAICDataset(**cp_cfg)
    else:
        from mmdet.datasets import builder
        return builder.build_dataset(cfg, default_args)
