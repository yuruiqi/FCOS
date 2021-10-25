import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor, make_udm_fcos_postprocessor
from .loss import make_fcos_loss_evaluator, make_udm_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, udm=None):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        self.udm = udm

        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        if cfg.MODEL.FCOS.UDM:
            self.cls_logits = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1
            )
        else:
            self.cls_logits = nn.Conv2d(
                in_channels, num_classes, kernel_size=3, stride=1,
                padding=1
            )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            if self.udm is None:
                logits.append(self.cls_logits(cls_tower))
            else:
                logits.append(self.udm(self.cls_logits(cls_tower)).squeeze(dim=1).permute(0,3,1,2))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        if cfg.MODEL.FCOS.UDM is not None:
            self.udm = UnsureDataLoss(cfg.MODEL.FCOS.UDM, device=cfg.MODEL.DEVICE)
            head = FCOSHead(cfg, in_channels, udm=self.udm)
            loss_evaluator = make_udm_fcos_loss_evaluator(cfg, self.udm)
            box_selector_test = make_udm_fcos_postprocessor(cfg, self.udm)
        else:
            self.udm = None
            head = FCOSHead(cfg, in_channels)
            loss_evaluator = make_fcos_loss_evaluator(cfg)
            box_selector_test = make_fcos_postprocessor(cfg)

        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)


#############################################
class UnsureDataLoss(nn.Module):
    def __init__(self, t, device, e=None, c=None):
        super().__init__()
        # avoid inf
        self.bias = torch.tensor([1e-7], dtype=torch.float32, device=device, requires_grad=False)

        self.t = nn.Parameter(torch.tensor(t, dtype=torch.float32), requires_grad=True)
        if e is None:
            self.e = torch.tensor([1, ] * len(t), dtype=torch.float32, device=device)
        else:
            # <1 to let t higher, or >1 to let t lower
            self.e = nn.Parameter(torch.tensor(e, dtype=torch.float32), requires_grad=True, device=device)

        if c is None:
            c = [0, ] * len(t)
        self.c = torch.tensor(c, dtype=torch.float32, device=device)

        self.num_class = len(t) + 1

    def forward(self, pred:torch.Tensor):
        """
        :param pred: (batch, *)
        :return:
        """
        confids = []
        for i in range(self.t.shape[0]+1):  # i means class
            if i == 0:
                confid_i = self.compute_confid(pred, t2=self.t[i], e2=self.e[i], c2=self.c[i])
            elif i == self.t.shape[0]:
                confid_i = self.compute_confid(pred, t1=self.t[i-1], e1=self.e[i-1], c1=self.c[i-1])
            else:
                confid_i = self.compute_confid(pred, self.t[i-1], self.t[i], self.e[i-1], self.e[i], self.c[i-1], self.c[i])
            confids.append(confid_i)

        confids = torch.stack(confids, dim=-1)
        return confids

    def loss(self, confids, label):
        all_class_loss = -torch.log(confids + self.bias)

        loss = 0
        for i in range(self.num_class):
            index = label.eq(i)
            i_loss = all_class_loss[...,i][index]
            if 0 not in torch.prod(torch.tensor(i_loss.shape)) != 0:
                loss += i_loss.mean()
        loss = loss/(self.num_class)

        return loss

    def inference(self, all_class_confid):
        """
        pred: (batch, 1)
        """
        confid, pred_out = all_class_confid.max(dim=-1)

        return pred_out, confid

    def compute_confid(self, x, t1=None, t2=None, e1=None, e2=None, c1=None, c2=None):
        zero = torch.tensor([0], dtype=x.dtype, device=x.device)
        if e1 is None:
            e1 = zero
        if e2 is None:
            e2 = zero
        if c1 is None:
            c1 = zero
        if c2 is None:
            c2 = zero

        if t1 is None:
            p_x = zero
        else:
            p_x = torch.sigmoid(-x + t1 + torch.log(e1 + self.bias) + c1)

        if t2 is None:
            p_x_add1 = torch.tensor([1], dtype=x.dtype, device=x.device)
        else:
            p_x_add1 = torch.sigmoid(-x + t2 + torch.log(e2 + self.bias) + c2)

        confid = p_x_add1 - p_x
        return confid

