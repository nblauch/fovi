import copy
import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torchvision.models import resnet18

from fovi.arch.knnresnet import KNNResNet
from fovi.arch.pretrained_resnet import (
    flattened_basic_blocks,
    load_torchvision_resnet_backbone,
    prep_fovi_resnet_finetuning,
)


def build_fovi_resnet(
    ref_frame_mult=2,
    stem_kernel_size=7,
    in_conv_stride=2,
    in_pool_stride=2,
):
    return KNNResNet(
        in_res=64,
        out_res=1,
        fov=16,
        cmf_a=0.5,
        style="isotropic",
        norm_type="batch",
        sample_cortex=False,
        device="cpu",
        auto_match_cart_resources=1,
        ref_frame_mult=ref_frame_mult,
        stem_kernel_size=stem_kernel_size,
        in_conv_stride=in_conv_stride,
        in_pool_stride=in_pool_stride,
    )


def expected_knn_weight(source_conv, target_conv, preserve_kernel_norm=False):
    weight = source_conv.weight.detach().transpose(-1, -2).flip(-1)
    if weight.shape[-2:] != (target_conv.ref_grid_size, target_conv.ref_grid_size):
        original_norm = weight.flatten(1).norm(dim=1)
        weight = F.interpolate(
            weight,
            size=(target_conv.ref_grid_size, target_conv.ref_grid_size),
            mode="bilinear",
            align_corners=True,
        )
        if preserve_kernel_norm:
            new_norm = weight.flatten(1).norm(dim=1).clamp_min(1e-12)
            weight = weight * (original_norm / new_norm).view(-1, 1, 1, 1)
    return weight.reshape(weight.shape[0], -1)


def make_finetune_cfg():
    return OmegaConf.create(
        {
            "pretrained_model": {
                "freeze_backbone": 1,
                "unfreeze_norm": 1,
                "lora": {
                    "layers": [-1, 0, 1, 2, 3],
                    "sublayers": ["conv1", "conv2", "downsample.0"],
                    "r": 4,
                    "alpha": 4,
                },
            }
        }
    )


class TestPretrainedResNetLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.source = resnet18(weights=None)
        bn_index = 1
        with torch.no_grad():
            for module in cls.source.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.weight.fill_(bn_index + 0.1)
                    module.bias.fill_(-bn_index - 0.2)
                    module.running_mean.fill_(bn_index + 0.3)
                    module.running_var.fill_(bn_index + 0.4)
                    module.num_batches_tracked.fill_(bn_index)
                    bn_index += 1

    def test_rf2_weight_and_batch_norm_mapping(self):
        target = build_fovi_resnet(ref_frame_mult=2)
        load_torchvision_resnet_backbone(target, self.source)

        self.assertEqual(target.conv1.ref_grid_size, 14)
        self.assertEqual(target.layer1[0].conv1.ref_grid_size, 6)
        torch.testing.assert_close(target.conv1.weight, expected_knn_weight(self.source.conv1, target.conv1))

        source_blocks = flattened_basic_blocks(self.source)
        target_blocks = flattened_basic_blocks(target)
        for source_block, target_block in zip(source_blocks, target_blocks):
            torch.testing.assert_close(
                target_block.conv1.weight,
                expected_knn_weight(source_block.conv1, target_block.conv1),
            )
            torch.testing.assert_close(
                target_block.conv2.weight,
                expected_knn_weight(source_block.conv2, target_block.conv2),
            )
            for source_bn, target_bn in (
                (source_block.bn1, target_block.norm1),
                (source_block.bn2, target_block.norm2),
            ):
                for state_name, source_value in source_bn.state_dict().items():
                    torch.testing.assert_close(target_bn.state_dict()[state_name], source_value)

            if source_block.downsample is not None:
                torch.testing.assert_close(
                    target_block.downsample[0].weight,
                    expected_knn_weight(source_block.downsample[0], target_block.downsample[0]),
                )

        for state_name, source_value in self.source.bn1.state_dict().items():
            torch.testing.assert_close(target.bn1.state_dict()[state_name], source_value)

    def test_native_and_downsampled_reference_frames(self):
        native = build_fovi_resnet(ref_frame_mult=1)
        load_torchvision_resnet_backbone(native, self.source)
        self.assertEqual(native.conv1.ref_grid_size, 7)
        self.assertEqual(native.layer1[0].conv1.ref_grid_size, 3)
        torch.testing.assert_close(native.conv1.weight, expected_knn_weight(self.source.conv1, native.conv1))

        downsampled = build_fovi_resnet(ref_frame_mult=0.5)
        load_torchvision_resnet_backbone(downsampled, self.source)
        self.assertEqual(downsampled.conv1.ref_grid_size, 4)
        self.assertEqual(downsampled.layer1[0].conv1.ref_grid_size, 2)
        self.assertEqual(downsampled.layer2[0].downsample[0].ref_grid_size, 1)
        torch.testing.assert_close(
            downsampled.layer1[0].conv1.weight,
            expected_knn_weight(self.source.layer1[0].conv1, downsampled.layer1[0].conv1),
        )

    def test_three_by_three_low_resolution_stem_transfer(self):
        target = build_fovi_resnet(
            ref_frame_mult=1,
            stem_kernel_size=3,
            in_conv_stride=1,
            in_pool_stride=1,
        )
        load_torchvision_resnet_backbone(target, self.source)

        self.assertEqual(target.conv1.k, 9)
        self.assertEqual(target.conv1.ref_grid_size, 3)
        self.assertGreater(len(target.layer1[0].out_coords), 230)
        torch.testing.assert_close(
            target.conv1.weight,
            expected_knn_weight(self.source.conv1, target.conv1),
        )

    def test_resampling_can_preserve_output_filter_norms(self):
        target = build_fovi_resnet(ref_frame_mult=2)
        load_torchvision_resnet_backbone(
            target, self.source, preserve_kernel_norm=True
        )

        expected = expected_knn_weight(
            self.source.layer1[0].conv1,
            target.layer1[0].conv1,
            preserve_kernel_norm=True,
        )
        torch.testing.assert_close(target.layer1[0].conv1.weight, expected)
        source_norm = self.source.layer1[0].conv1.weight.flatten(1).norm(dim=1)
        target_norm = target.layer1[0].conv1.weight.flatten(1).norm(dim=1)
        torch.testing.assert_close(target_norm, source_norm)

    def test_partial_finetuning_and_all_norms(self):
        target = build_fovi_resnet(ref_frame_mult=1)
        cfg = OmegaConf.create(
            {
                "pretrained_model": {
                    "freeze_backbone": 1,
                    "unfreeze_norm": 0,
                    "unfreeze_all_norms": 1,
                    "unfreeze_layers": [-1, 6, 7],
                    "lora": None,
                }
            }
        )
        prep_fovi_resnet_finetuning(target, cfg, device="cpu")

        self.assertTrue(target.conv1.weight.requires_grad)
        self.assertTrue(target.bn1.weight.requires_grad)
        self.assertFalse(target.layer3[-1].conv2.weight.requires_grad)
        self.assertTrue(target.layer4[0].conv1.weight.requires_grad)
        self.assertTrue(target.layer4[1].conv2.weight.requires_grad)
        for module in target.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                self.assertTrue(module.weight.requires_grad)
                self.assertTrue(module.bias.requires_grad)

    def test_lora_trainability_gradients_frozen_stats_and_checkpoint(self):
        target = build_fovi_resnet(ref_frame_mult=2)
        load_torchvision_resnet_backbone(target, self.source)
        prep_fovi_resnet_finetuning(target, make_finetune_cfg(), device="cpu")

        blocks = flattened_basic_blocks(target)
        lora_modules = [target.conv1]
        for block in blocks[:4]:
            lora_modules.extend([block.conv1, block.conv2])
            if block.downsample is not None:
                lora_modules.append(block.downsample[0])

        for module in lora_modules:
            self.assertTrue(hasattr(module, "parametrizations"))
            self.assertFalse(module.parametrizations.weight.original.requires_grad)
            self.assertTrue(module.parametrizations.weight[0].A.requires_grad)
            self.assertTrue(module.parametrizations.weight[0].B.requires_grad)

        self.assertTrue(target.layer4[-1].norm2.weight.requires_grad)
        self.assertTrue(target.layer4[-1].norm2.bias.requires_grad)
        self.assertFalse(target.layer4[-1].conv2.weight.requires_grad)

        target.eval()
        running_mean = target.layer4[-1].norm2.running_mean.detach().clone()
        x = torch.randn(1, 3, len(target.in_coords))
        target(x).sum().backward()

        for module in lora_modules:
            self.assertIsNotNone(module.parametrizations.weight[0].A.grad)
            self.assertIsNotNone(module.parametrizations.weight[0].B.grad)
        self.assertIsNotNone(target.layer4[-1].norm2.weight.grad)
        torch.testing.assert_close(target.layer4[-1].norm2.running_mean, running_mean)

        checkpoint = copy.deepcopy(target.state_dict())
        restored = build_fovi_resnet(ref_frame_mult=2)
        prep_fovi_resnet_finetuning(restored, make_finetune_cfg(), device="cpu")
        restored.load_state_dict(checkpoint)
        for key, value in checkpoint.items():
            torch.testing.assert_close(restored.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
