import torch.nn as nn
from models.layer import *
from models.temporal_coding import make_time_scales

cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG, self).__init__()
        self.init_channels = 3
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.loss = 0
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        if num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(IF())
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def set_T(self, T):
        self.T = T
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim)):
                module.T = T
        return

    def set_L(self, L):
        for module in self.modules():
            if isinstance(module, IF):
                module.L = L
        return

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
        return out

class VGG_woBN(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG_woBN, self).__init__()
        self.init_channels = 3
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        if num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(IF())
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def set_T(self, T):
        self.T = T
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim)):
                module.T = T
        return

    def set_L(self, L):
        for module in self.modules():
            if isinstance(module, IF):
                module.L = L
        return

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
        return out


class VGG_Signed(nn.Module):
    """VGG with SNM Signed Spike + Memory neurons"""
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG_Signed, self).__init__()
        self.init_channels = 3
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.loss = 0
        self.coding_mode = "rate"
        self.refinement_schedule = "geometric"
        self.refinement_ratio = 2.0
        self.refinement_positive_margin = 0.5
        self.refinement_negative_margin = 0.5
        self.refinement_r0_mode = "credit_only"
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        if num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                SignedIF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                SignedIF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096),
                SignedIF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                SignedIF(),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(SignedIF())  # Use SignedIF instead of IF
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def set_T(self, T):
        self.T = T
        for module in self.modules():
            if isinstance(module, (SignedIF, ExpandTemporalDim)):
                module.T = T
            if isinstance(module, SignedIF):
                module.init_mem()
                module.time_scales = None
                module._time_scale_cache_key = None
                module.reset_stats()
        return

    def set_thresh(self, thresh):
        import torch
        for module in self.modules():
            if isinstance(module, SignedIF):
                module.thresh.data = torch.tensor([thresh])
                module.neg_thresh.data = torch.tensor([-thresh])
        return

    # =================== [新增] 消融实验控制方法 ===========================
    def set_signed(self, enabled):
        """开/关所有 SignedIF 层的负脉冲（SNM）"""
        for m in self.modules():
            if isinstance(m, SignedIF):
                m.enable_signed = enabled

    def set_r0(self, enabled):
        """开/关所有 SignedIF 层的 R0 无负债规则"""
        for m in self.modules():
            if isinstance(m, SignedIF):
                m.enable_r0 = enabled

    def set_snm_negative_margin(self, margin):
        """Set one residual SNM dead-band for every signed activation."""
        for m in self.modules():
            if isinstance(m, SignedIF):
                m.set_snm_negative_margin(margin)

    def set_snm_mode(self, mode, start=1.25, end=0.5, reference=8.0):
        """Set standard SNM or the horizon-annealed variant."""
        for m in self.modules():
            if isinstance(m, SignedIF):
                m.set_snm_mode(
                    mode, start=start, end=end, reference=reference
                )

    def set_ftbc_mode(self, mode):
        """Set the FTBC representation used by every SignedIF layer."""
        for m in self.modules():
            if isinstance(m, SignedIF):
                m.set_ftbc_mode(mode)

    def set_coding_mode(
        self,
        mode,
        schedule="geometric",
        ratio=2.0,
        positive_margin=0.5,
        negative_margin=0.5,
        r0_mode="credit_only",
    ):
        """Set one temporal coding rule for all SignedIF layers."""
        self.coding_mode = mode
        self.refinement_schedule = schedule
        self.refinement_ratio = float(ratio)
        self.refinement_positive_margin = float(positive_margin)
        self.refinement_negative_margin = float(negative_margin)
        self.refinement_r0_mode = r0_mode
        for module in self.modules():
            if isinstance(module, SignedIF):
                module.set_coding_mode(
                    mode,
                    schedule=schedule,
                    ratio=ratio,
                    positive_margin=positive_margin,
                    negative_margin=negative_margin,
                    r0_mode=r0_mode,
                )

    def apply_temporal_readout(self, output):
        """Apply refinement scales before the existing temporal mean."""
        if self.coding_mode == "rate":
            return output
        scales = make_time_scales(
            self.T,
            mode=self.refinement_schedule,
            ratio=self.refinement_ratio,
            device=output.device,
            dtype=output.dtype,
        )
        shape = [self.T] + [1] * (output.dim() - 1)
        return output * scales.view(*shape)

    def reset_all_bias(self):
        """清零所有 SignedIF 层的 FTBC 时间步偏置"""
        for m in self.modules():
            if isinstance(m, SignedIF):
                m.reset_bias()
    # =====================================================================

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
            out = self.apply_temporal_readout(out)
        return out


def vgg16(num_classes, dropout=0.):
    return VGG('VGG16', num_classes, dropout)

def vgg16_wobn(num_classes, dropout=0.1):
    return VGG_woBN('VGG16', num_classes, dropout)

def vgg16_signed(num_classes, dropout=0.):
    """VGG16 with SNM Signed Spike + Memory neurons"""
    return VGG_Signed('VGG16', num_classes, dropout)

def vgg19(num_classes, dropout):
    return VGG('VGG19', num_classes, dropout)

