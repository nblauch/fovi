"""Tests for the retinal-sampling front-end (fovi/sensing/retina.py, policies.py).

Covers the front-end optimizations:
- sample-then-augment fast path for pre-warp transforms (bit-exact parity vs the reference
  full-image path, identical RNG stream consumption, out-of-bounds zero-padding semantics);
- vectorized near-center fixation sampling (bitwise RNG parity with the per-sample loop);
- pure-tensor fixation-argument handling (numpy / tuple / list / scalar / CPU / CUDA tensor);
- explicit-fixations vs fresh-sampling consistency of the fixation policy;
- train/eval mode behavior.

Run with: python -m unittest tests.test_retinal_sampling -v
Device selection: first CUDA device by default; override with FOVI_TEST_DEVICE=<index>.
"""

import os
import unittest

import numpy as np
import torch

from fovi.sensing.retina import RetinalTransform
from fovi.sensing.policies import MultiRandomSaccadePolicy
from fovi.utils.fastaugs import transforms as fastT
from fovi.utils.std_transforms import get_std_transforms

_HAVE_CUDA = torch.cuda.is_available()


def _device():
    index = int(os.environ.get("FOVI_TEST_DEVICE", "0"))
    return torch.device("cuda", index)


def _make_retinal_transform(device, sampler="grid_nn", with_transforms=True,
                            color_jitter=1, gray=1, blur=0):
    pre = post = None
    if with_transforms:
        _, pre, post = get_std_transforms(
            "pre_warp", 1, color_jitter, gray, blur, str(device), torch.float32,
            pointcloud_mode=True)
    rt = RetinalTransform(
        resolution=64, start_res=256, fov=16.0, cmf_a=0.5, style="isotropic",
        sampler=sampler, fixation_size=256, device=str(device),
        pre_transforms=pre, post_transforms=post, auto_match_cart_resources=0)
    return rt


def _make_inputs(device, batch=16, seed=1234):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.rand(batch, 3, 256, 256, generator=g, device=device)
    # spread fixations widely so some sampled points fall out of bounds
    fix_loc = torch.rand(batch, 2, generator=g, device=device) * 0.9 + 0.05
    fix_size = torch.tensor([[256, 256]]).repeat(batch, 1)
    return x, fix_loc, fix_size


@unittest.skipUnless(_HAVE_CUDA, "requires CUDA")
class TestFastPreTransformParity(unittest.TestCase):
    """The sample-then-augment fast path must be bit-exact vs the reference path and must
    consume the CUDA RNG stream identically."""

    def _run_both(self, rt, x, fix_loc, fix_size, seed=999):
        device = x.device
        torch.manual_seed(seed)
        cuda_state = torch.cuda.get_rng_state(device)

        rt.fast_pre_transforms = False
        out_ref = rt(x, fix_loc, fixation_size=fix_size)
        state_ref = torch.cuda.get_rng_state(device)

        torch.cuda.set_rng_state(cuda_state, device)
        rt.fast_pre_transforms = True
        out_fast = rt(x, fix_loc, fixation_size=fix_size)
        state_fast = torch.cuda.get_rng_state(device)
        return out_ref, out_fast, state_ref, state_fast

    def test_train_mode_bit_exact(self):
        device = _device()
        rt = _make_retinal_transform(device)
        rt.train()
        self.assertTrue(rt._fast_pre_transforms_supported())
        x, fix_loc, fix_size = _make_inputs(device)
        out_ref, out_fast, state_ref, state_fast = self._run_both(rt, x, fix_loc, fix_size)
        self.assertEqual((out_ref - out_fast).abs().max().item(), 0.0)
        self.assertTrue(torch.equal(state_ref, state_fast),
                        "fast path must not change downstream RNG draws")

    def test_train_mode_bit_exact_multiple_seeds(self):
        device = _device()
        rt = _make_retinal_transform(device)
        rt.train()
        x, fix_loc, fix_size = _make_inputs(device, seed=7)
        for seed in (0, 1, 42):
            out_ref, out_fast, _, _ = self._run_both(rt, x, fix_loc, fix_size, seed=seed)
            self.assertEqual((out_ref - out_fast).abs().max().item(), 0.0,
                             f"parity failure at seed {seed}")

    def test_out_of_bounds_points_are_zero(self):
        """Reference semantics: grid_sample zero-pads AFTER the pre-warp transforms, so OOB
        points must be exactly zero (not transform-of-zero) in both paths."""
        device = _device()
        rt = _make_retinal_transform(device)
        rt.train()
        batch = 8
        g = torch.Generator(device=device).manual_seed(3)
        x = torch.rand(batch, 3, 256, 256, generator=g, device=device)
        # fixation at the far corner: much of the grid lands outside the image
        fix_loc = torch.full((batch, 2), 0.02, device=device)
        fix_size = torch.tensor([[256, 256]]).repeat(batch, 1)
        out_ref, out_fast, _, _ = self._run_both(rt, x, fix_loc, fix_size)
        self.assertEqual((out_ref - out_fast).abs().max().item(), 0.0)
        # sanity: OOB points exist and are exactly zero in both
        fl = rt._check_fix_loc(fix_loc, batch)
        fs = rt._check_fixation_size(fix_size, batch)
        grid = rt.sampler._transform_fix_grid(x.shape[-2:], fl, fs)
        mask = torch.nn.functional.grid_sample(
            rt._padding_mask_src(x), grid, mode="nearest", align_corners=False).squeeze(2)
        n_oob = int((mask == 0).sum().item())
        self.assertGreater(n_oob, 0)
        self.assertEqual(out_fast.transpose(0, 1)[:, mask.squeeze(1) == 0].abs().max().item(), 0.0)

    def test_gray_only_and_jitter_only(self):
        device = _device()
        for cj, gray in ((1, 0), (0, 1)):
            rt = _make_retinal_transform(device, color_jitter=cj, gray=gray)
            rt.train()
            x, fix_loc, fix_size = _make_inputs(device, seed=11)
            out_ref, out_fast, state_ref, state_fast = self._run_both(rt, x, fix_loc, fix_size)
            self.assertEqual((out_ref - out_fast).abs().max().item(), 0.0,
                             f"parity failure for color_jitter={cj}, gray={gray}")
            self.assertTrue(torch.equal(state_ref, state_fast))

    def test_eval_mode_unaffected(self):
        device = _device()
        rt = _make_retinal_transform(device)
        rt.eval()
        x, fix_loc, fix_size = _make_inputs(device)
        out1 = rt(x, fix_loc, fixation_size=fix_size)
        rt.fast_pre_transforms = False
        out2 = rt(x, fix_loc, fixation_size=fix_size)
        rt.fast_pre_transforms = True
        self.assertTrue(torch.equal(out1, out2))

    def test_bilinear_sampler_not_eligible(self):
        """Pointwise transforms do not commute with bilinear interpolation; the fast path
        must decline and fall back to the reference path."""
        device = _device()
        rt = _make_retinal_transform(device, sampler="grid_bilinear")
        rt.train()
        self.assertFalse(rt._fast_pre_transforms_supported())

    def test_unsupported_transform_not_eligible(self):
        device = _device()
        rt = _make_retinal_transform(device, blur=1)  # blur is spatial, not pointwise
        rt.train()
        self.assertFalse(rt._fast_pre_transforms_supported())
        # and the fallback still runs
        x, fix_loc, fix_size = _make_inputs(device, batch=4)
        out = rt(x, fix_loc, fixation_size=fix_size)
        self.assertEqual(out.shape[0], 4)

    def test_spatial_pre_transform_routes_to_reference_path(self):
        """A spatial transform in pre_transforms must never take the fast path (it does not
        commute with sampling): the guard must decline and the forward must produce exactly
        the reference full-image-transform-then-sample result. The allowlist is strict by
        exact type: probability-gated containers (RandomApply) and subclasses of supported
        classes (whose overridden behavior the fast path would not replicate) must also be
        rejected."""
        device = _device()
        rt = _make_retinal_transform(device)
        rt.pre_transforms.transforms.append(fastT.RandomHorizontalFlip())  # spatial
        rt.train()
        self.assertFalse(rt._fast_pre_transforms_supported())
        # and the forward routes to the reference path: full-image transforms, then sampling
        x, fix_loc, fix_size = _make_inputs(device, batch=8)
        torch.manual_seed(0)
        cuda_state = torch.cuda.get_rng_state(device)
        out = rt(x, fix_loc, fixation_size=fix_size)
        torch.manual_seed(0)
        torch.cuda.set_rng_state(cuda_state, device)
        fl = rt._check_fix_loc(fix_loc, x.shape[0])
        fs = rt._check_fixation_size(fix_size, x.shape[0])
        ref = rt.sampler(rt.pre_transforms(x.clone()), fix_loc=fl, fixation_size=fs)
        self.assertTrue(torch.equal(out, ref.to(rt.dtype)))
        # non-Compose container exposing `.transforms` (different call semantics): rejected
        rt2 = _make_retinal_transform(device)
        rt2.pre_transforms = fastT.RandomApply(list(rt2.pre_transforms.transforms), p=0.5)
        rt2.train()
        self.assertFalse(rt2._fast_pre_transforms_supported())
        # subclass of a supported class: rejected (exact-type allowlist)
        class _SubGrayscale(fastT.RandomGrayscale):
            pass
        rt3 = _make_retinal_transform(device)
        rt3.pre_transforms = fastT.Compose([_SubGrayscale(p=0.2)])
        rt3.train()
        self.assertFalse(rt3._fast_pre_transforms_supported())

    def test_input_not_mutated(self):
        device = _device()
        rt = _make_retinal_transform(device)
        rt.train()
        x, fix_loc, fix_size = _make_inputs(device, batch=4)
        x_before = x.clone()
        rt(x, fix_loc, fixation_size=fix_size)
        self.assertTrue(torch.equal(x, x_before))


@unittest.skipUnless(_HAVE_CUDA, "requires CUDA")
class TestFixationArgFormats(unittest.TestCase):
    """_check_fix_loc / _check_fixation_size must accept the same argument formats as before
    (None / scalar / tuple / list / numpy / CPU tensor / CUDA tensor) and produce the same
    values, now without host round-trips for tensors."""

    def setUp(self):
        self.device = _device()
        self.rt = _make_retinal_transform(self.device, with_transforms=False)
        self.batch = 6

    def _assert_loc(self, fix_loc, expected):
        out = self.rt._check_fix_loc(fix_loc, self.batch)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (self.batch, 2))
        self.assertEqual(out.dtype, self.rt.dtype)
        self.assertEqual(out.device.type, "cuda")
        expected = torch.as_tensor(expected, dtype=out.dtype)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    def test_fix_loc_variants(self):
        b = self.batch
        self._assert_loc(None, torch.tensor([0.5, 0.5]).expand(b, 2))
        self._assert_loc(0.25, torch.full((b, 2), 0.25))
        self._assert_loc((0.3, 0.7), torch.tensor([0.3, 0.7]).expand(b, 2))
        self._assert_loc([0.3, 0.7], torch.tensor([0.3, 0.7]).expand(b, 2))
        self._assert_loc(np.array([0.3, 0.7]), torch.tensor([0.3, 0.7]).expand(b, 2))
        per_batch = np.linspace(0.1, 0.9, 2 * b).reshape(b, 2)
        self._assert_loc(per_batch, torch.tensor(per_batch, dtype=torch.float32))
        t = torch.tensor(per_batch, dtype=torch.float32)
        self._assert_loc(t, t)                       # CPU tensor
        self._assert_loc(t.to(self.device), t)       # CUDA tensor
        self._assert_loc(torch.tensor([0.3, 0.7]), torch.tensor([0.3, 0.7]).expand(b, 2))

    def _assert_size(self, fixation_size, expected):
        out = self.rt._check_fixation_size(fixation_size, self.batch)
        arr = out.cpu().numpy() if isinstance(out, torch.Tensor) else np.asarray(out)
        self.assertEqual(arr.shape, (self.batch, 2))
        np.testing.assert_array_equal(arr, np.asarray(expected))

    def test_fixation_size_variants(self):
        b = self.batch
        full = np.full((b, 2), 256)
        self._assert_size(None, full)
        self._assert_size(256, full)
        self._assert_size("none", full)
        self._assert_size((128, 64), np.tile([128, 64], (b, 1)))
        self._assert_size([128, 64], np.tile([128, 64], (b, 1)))
        self._assert_size(np.array([128, 64]), np.tile([128, 64], (b, 1)))
        per_batch = np.arange(b * 2).reshape(b, 2) + 10
        self._assert_size(per_batch, per_batch)
        t = torch.tensor(per_batch)
        self._assert_size(t, per_batch)                     # CPU tensor (B,2)
        self._assert_size(t.to(self.device), per_batch)     # CUDA tensor (B,2)
        self._assert_size(torch.tensor([128, 64]), np.tile([128, 64], (b, 1)))
        per_batch_1d = np.arange(b) + 50
        self._assert_size(per_batch_1d, np.stack([per_batch_1d] * 2, axis=1))
        self._assert_size(torch.tensor(per_batch_1d), np.stack([per_batch_1d] * 2, axis=1))

    def test_forward_equivalence_across_formats(self):
        """The full forward must produce identical outputs for equivalent fix_loc formats."""
        rt = self.rt
        rt.eval()
        g = torch.Generator(device=self.device).manual_seed(5)
        x = torch.rand(self.batch, 3, 256, 256, generator=g, device=self.device)
        loc = np.tile([0.4, 0.6], (self.batch, 1))
        outs = [
            rt(x, np.array([0.4, 0.6])),
            rt(x, (0.4, 0.6)),
            rt(x, [0.4, 0.6]),
            rt(x, torch.tensor([0.4, 0.6], dtype=torch.float32)),
            rt(x, torch.tensor(loc, dtype=torch.float32)),
            rt(x, torch.tensor(loc, dtype=torch.float32).to(self.device)),
        ]
        for i, out in enumerate(outs[1:], start=1):
            self.assertTrue(torch.equal(outs[0], out), f"format {i} differs")


@unittest.skipUnless(_HAVE_CUDA, "requires CUDA")
class TestVectorizedFixationSampling(unittest.TestCase):
    """get_random_nearcenter_fixations_batch must consume the global numpy RNG identically
    to n sequential get_random_nearcenter_fixation calls (bitwise-equal results)."""

    def setUp(self):
        self.device = _device()
        rt = _make_retinal_transform(self.device, with_transforms=False)
        self.policy = MultiRandomSaccadePolicy(
            rt, n_fixations=4, norm_dist_from_center=0.25, crop_area_range=[0.3, 1.0])
        self.policy.train()

    def _sequential(self, n, scale, ratio, ndfc):
        fixations, sizes = [], []
        for _ in range(n):
            f, s = self.policy.get_random_nearcenter_fixation(
                256, 256, scale=scale, ratio=ratio, normalized_dist_from_center=ndfc)
            fixations.append(f)
            sizes.append(s)
        return (torch.tensor(fixations, dtype=self.policy.dtype, device=self.policy.device),
                torch.tensor(sizes))

    def test_bitwise_rng_parity(self):
        cases = [
            ([0.3, 1.0], 1),          # area range, scalar ratio (audited configs)
            ([1.0, 1.0], 1),          # degenerate area range (benchmark configs)
            (0.7, 1),                 # scalar area
            ([0.3, 1.0], [3 / 4, 4 / 3]),  # aspect variation enabled
            (0.7, [3 / 4, 4 / 3]),
        ]
        for scale, ratio in cases:
            np.random.seed(123)
            f_seq, s_seq = self._sequential(33, scale, ratio, 0.25)
            np.random.seed(123)
            f_vec, s_vec = self.policy.sample_fixations(
                (256, 256), n=33, area_range=scale, ratio=ratio, norm_dist_from_center=0.25)
            self.assertTrue(torch.equal(f_seq, f_vec), f"fixations differ for {scale}/{ratio}")
            self.assertTrue(torch.equal(s_seq, s_vec), f"sizes differ for {scale}/{ratio}")

    def test_rng_stream_end_state(self):
        """The vectorized path must leave the numpy RNG in the same state."""
        np.random.seed(9)
        self._sequential(17, [0.5, 1.0], 1, 0.25)
        state_seq = np.random.get_state()[1]
        np.random.seed(9)
        self.policy.sample_fixations((256, 256), n=17, area_range=[0.5, 1.0], ratio=1,
                                     norm_dist_from_center=0.25)
        state_vec = np.random.get_state()[1]
        np.testing.assert_array_equal(state_seq, state_vec)


@unittest.skipUnless(_HAVE_CUDA, "requires CUDA")
class TestPolicyFrontEnd(unittest.TestCase):
    """Front-end (fixation policy + retinal transform) consistency tests."""

    def _make_policy(self):
        device = _device()
        rt = _make_retinal_transform(device)
        policy = MultiRandomSaccadePolicy(
            rt, n_fixations=4, norm_dist_from_center=0.25, crop_area_range=[1.0, 1.0])
        return policy, device

    def test_fresh_vs_explicit_fixations_consistency(self):
        """Passing the fixations drawn by a fresh-sampling call back in as explicit
        fixations must reproduce the same sampled outputs (at the same RNG state)."""
        policy, device = self._make_policy()
        policy.train()
        g = torch.Generator(device=device).manual_seed(21)
        x = torch.rand(8, 3, 256, 256, generator=g, device=device)

        np.random.seed(42)
        torch.manual_seed(42)
        cuda_state = torch.cuda.get_rng_state(device)
        fresh = policy(x)

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.set_rng_state(cuda_state, device)
        explicit = policy(x, fixations=list(fresh["fixations"].unbind(dim=1)))

        self.assertTrue(torch.equal(fresh["x_fixs"], explicit["x_fixs"]))
        self.assertTrue(torch.equal(fresh["fixations"], explicit["fixations"]))

    def test_frontend_fast_vs_reference_end_to_end(self):
        """Whole 4-fixation front-end, fast vs reference path: bit-exact at fixed seeds."""
        policy, device = self._make_policy()
        policy.train()
        g = torch.Generator(device=device).manual_seed(22)
        x = torch.rand(8, 3, 256, 256, generator=g, device=device)

        results = {}
        for fast in (False, True):
            policy.retinal_transform.fast_pre_transforms = fast
            np.random.seed(7)
            torch.manual_seed(7)
            torch.cuda.manual_seed_all(7)
            results[fast] = policy(x)
        policy.retinal_transform.fast_pre_transforms = True
        self.assertTrue(torch.equal(results[False]["x_fixs"], results[True]["x_fixs"]))
        self.assertTrue(torch.equal(results[False]["fixations"], results[True]["fixations"]))

    def test_eval_mode_deterministic_sampler_path(self):
        policy, device = self._make_policy()
        policy.eval()
        g = torch.Generator(device=device).manual_seed(23)
        x = torch.rand(4, 3, 256, 256, generator=g, device=device)
        fix = torch.full((4, 2), 0.5, device=device)
        out1 = policy.retinal_transform(x, fix)
        out2 = policy.retinal_transform(x, fix)
        self.assertTrue(torch.equal(out1, out2))


class TestNormalizeStatsDtype(unittest.TestCase):
    """NormalizeGPU must hold float32 stats: float64 stats promote the whole normalize
    computation of a float32 image to float64 (very slow on consumer GPUs)."""

    def test_float32_stats(self):
        from fovi.utils.fastaugs import transforms as fastT
        norm = fastT.NormalizeGPU(mean=np.array([0.485, 0.456, 0.406]),
                                  std=np.array([0.229, 0.224, 0.225]), device="cpu")
        self.assertEqual(norm.mean.dtype, torch.float32)
        self.assertEqual(norm.std.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
