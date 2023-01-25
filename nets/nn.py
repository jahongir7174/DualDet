import torch
from mmcv.runner import BaseModule
from mmdet.models import backbones, builder, detectors, necks, roi_heads
from torch.nn.functional import interpolate


def build_detector(cfg, train_cfg=None, test_cfg=None):
    args = dict(train_cfg=train_cfg, test_cfg=test_cfg)
    return builder.MODELS.build(cfg, default_args=args)


class DualModule(backbones.SwinTransformer):
    def _freeze_stages(self):
        if self.frozen_stages >= 0 and hasattr(self, 'patch_embed'):
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.use_abs_pos_embed:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.drop_after_pos.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.stages[i]
                if m is None:
                    continue
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def del_layers(self, del_stages):
        if del_stages >= 0:
            del self.patch_embed

        if del_stages >= 1 and self.use_abs_pos_embed:
            del self.absolute_pos_embed

        for i in range(0, del_stages - 1):
            self.stages[i] = None

    def forward(self, x, y1=None, y2=None):
        x1 = []
        x2 = []
        if hasattr(self, 'patch_embed'):
            x, (h_down, w_down) = self.patch_embed(x)

            x = self.drop_after_pos(x)

            x2.append((x, h_down, w_down))
        else:
            x, h_down, w_down = y2[0]

        for i in range(len(self.stages)):
            layer = self.stages[i]
            if layer is None:
                y, h, w, x, h_down, w_down = y2[i + 1]
            else:
                if y1 is not None:
                    x = x + y1[i]
                x, (h_down, w_down), y, (h, w) = layer(x, (h_down, w_down))
            x2.append((y, h, w, x, h_down, w_down))

            if i in (0, 1, 2, 3):
                out = getattr(self, f'norm{i}')(y)
                out = out.view(-1, h, w, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                x1.append(out)

        return tuple(x1), x2

    def train(self, mode=True):
        """
        Convert the model into training mode while keep layers frozen.
        """
        super(DualModule, self).train(mode)
        self._freeze_stages()


@builder.MODELS.register_module()
class DualViT(BaseModule):
    def __init__(self, k=2, embed_dims=96, **kwargs):
        super(DualViT, self).__init__()
        self.dual_modules = torch.nn.ModuleList()

        for i in range(k):
            dual_module = DualModule(embed_dims=embed_dims, **kwargs)
            if i > 0:
                dual_module.del_layers(1)
            self.dual_modules.append(dual_module)

        self.num_layers = len(self.dual_modules[0].stages)

        in_channels = [embed_dims * 2 ** i for i in range(self.num_layers)]

        self.dual_layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            layers = torch.nn.ModuleList()
            if i >= 0:
                for j in range(4 - i):
                    if in_channels[i + j] != in_channels[i]:
                        layer = torch.nn.Conv2d(in_channels[i + j], in_channels[i], 1)
                    else:
                        layer = torch.nn.Identity()
                    layers.append(layer)
            self.dual_layers.append(layers)

    def _freeze_stages(self):
        for m in self.dual_modules:
            m._freeze_stages()

    def init_weights(self):
        """
        Initialize the weights in backbone.
        """
        from mmcv.cnn import constant_init

        for layers in self.dual_layers:
            for m in layers:
                constant_init(m, 0)

        for m in self.dual_modules:
            m.init_weights()

    @staticmethod
    def spatial_interpolate(x, h, w):
        b, c = x.shape[:2]
        if h != x.shape[2] or w != x.shape[3]:
            x = interpolate(x, size=(h, w), mode='nearest')
        return x.view(b, c, -1).permute(0, 2, 1).contiguous()  # B, T, C

    def _get_dual_feats(self, y1, y2):
        outputs = []
        h_down, w_down = y2[0][-2:]
        for i in range(self.num_layers):
            x = 0
            if i >= 0:
                for j in range(4 - i):
                    y = self.dual_layers[i][j](y1[j + i])
                    y = self.spatial_interpolate(y, h_down, w_down)
                    x += y
            outputs.append(x)
            h_down, w_down = y2[i + 1][-2:]
        return outputs

    def forward(self, x):
        outputs = []
        for i, module in enumerate(self.dual_modules):
            if i == 0:
                x1, x2 = module(x)
            else:
                x1, x2 = module(x, y, x2)

            outputs.append(x1)

            if i < len(self.dual_modules) - 1:
                y = self._get_dual_feats(outputs[-1], x2)
        return tuple(outputs)

    def train(self, mode=True):
        """
        Convert the model into training mode while keep layers frozen.
        """
        super(DualViT, self).train(mode)
        for m in self.dual_modules:
            m.train(mode=mode)
        self._freeze_stages()


@builder.MODELS.register_module()
class DualFPN(necks.FPN):
    def forward(self, inputs):
        if not isinstance(inputs[0], (list, tuple)):
            inputs = [inputs]

        if self.training:
            outputs = []
            for x in inputs:
                outputs.append(super().forward(x))
            return outputs
        else:
            return super().forward(inputs[-1])


@builder.MODELS.register_module()
class BoxHead(roi_heads.ConvFCBBoxHead):
    def __init__(self, *args, **kwargs):
        super().__init__(0, 2, *args, **kwargs)
        self.relu = torch.nn.SiLU(inplace=True)


@builder.MODELS.register_module()
class DualDet(detectors.FasterRCNN):
    @staticmethod
    def update_loss(x, y, w):
        for k, v in y.items():
            if 'loss' in k:
                if isinstance(v, list) or isinstance(v, tuple):
                    if k not in x:
                        x[k] = sum([i * w for i in v])
                    else:
                        x[k] += sum([i * w for i in v])
                else:
                    if k not in x:
                        x[k] = v * w
                    else:
                        x[k] += v * w
        return x

    def forward_train(self,
                      img, img_metas,
                      gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None, proposals=None, **kwargs):
        x_list = self.extract_feat(img)

        if not isinstance(x_list[0], (list, tuple)):
            x_list = [x_list]
            loss_weights = [1]
        else:
            loss_weights = [0.5] + [1] * (len(x_list) - 1)  # Reference CBNet paper

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            for loss_weight, x in zip(loss_weights, x_list):
                rpn_losses, proposal_list = self.rpn_head.forward_train(x,
                                                                        img_metas,
                                                                        gt_bboxes,
                                                                        gt_labels=None,
                                                                        gt_bboxes_ignore=gt_bboxes_ignore,
                                                                        proposal_cfg=proposal_cfg)
                losses = self.update_loss(losses, rpn_losses, loss_weight)
        else:
            proposal_list = proposals

        # ROI forward and loss
        for loss_weight, x in zip(loss_weights, x_list):
            roi_losses = self.roi_head.forward_train(x, img_metas,
                                                     proposal_list,
                                                     gt_bboxes, gt_labels,
                                                     gt_bboxes_ignore, gt_masks, **kwargs)
            losses = self.update_loss(losses, roi_losses, loss_weight)
        return losses
