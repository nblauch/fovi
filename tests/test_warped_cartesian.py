"""FoV geometry and native warped-Cartesian sensor tests."""

import math
import unittest

import torch

from fovi.arch.knn import KNNConvLayer, KNNGetterLayer
from fovi.sensing.coords import (
    SamplingCoords,
    _inverse_warped_cartesian,
    _warped_cartesian_radius_normalizer,
    find_desired_res,
    get_warped_cartesian_sampling_coords,
    num_sampling_coords,
)
from fovi.sensing.retina import RetinalTransform


class TestFoVGeometry(unittest.TestCase):
    def test_isotropic_square_filters_corner_reaching_manifold(self):
        coords = SamplingCoords(
            16.0, 0.5, 16, style="isotropic", fov_type="square",
            isotropic_plotting_type="warp")
        square_half_extent = 1.0

        self.assertTrue(torch.all(coords.valid_mask))
        self.assertTrue(torch.all(
            coords.cartesian.abs() <= square_half_extent + 1e-6))
        self.assertAlmostEqual(
            coords.polar[:, 0].max().item(), math.sqrt(2.0), places=5)
        corners = torch.tensor(
            [[-square_half_extent, -square_half_extent],
             [-square_half_extent, square_half_extent],
             [square_half_extent, -square_half_extent],
             [square_half_extent, square_half_extent]],
            dtype=coords.cartesian.dtype)
        for corner in corners:
            self.assertTrue(torch.any(torch.all(torch.isclose(
                coords.cartesian, corner[None], atol=1e-6), dim=1)))
        self.assertEqual(
            len(coords),
            num_sampling_coords(
                16.0, 0.5, 16, style="isotropic", fov_type="square"))

    def test_square_resource_matching_counts_only_retained_points(self):
        resolution, count = find_desired_res(
            16.0, 0.5, 64, "isotropic", fov_type="square",
            force_less_than=True, quiet=True)
        coords = SamplingCoords(
            16.0, 0.5, resolution, style="isotropic", fov_type="square")

        self.assertEqual(count, len(coords))
        self.assertLessEqual(count, 64)
        self.assertEqual(
            count,
            num_sampling_coords(
                16.0, 0.5, resolution, "isotropic", fov_type="square"))

    def test_small_square_fixed_count_grid_has_spatial_padding(self):
        coords = SamplingCoords(
            16.0, 0.5, 2, style="isotropic_fixn", fov_type="square")

        self.assertEqual(len(coords), 4)
        self.assertGreater(len(coords.cartesian_pad_coords), 0)

    def test_logpolar_square_keeps_grid_and_masks_outside_square(self):
        coords = SamplingCoords(
            16.0, 0.5, 16, style="logpolar_as_grid", fov_type="square")

        self.assertEqual(len(coords), 16 ** 2)
        self.assertEqual(coords.valid_mask.shape, (16 ** 2,))
        square_half_extent = 1.0
        expected = (
            coords.cartesian.abs().amax(dim=1)
            <= square_half_extent + 2e-6)
        torch.testing.assert_close(coords.valid_mask, expected)
        self.assertTrue(torch.any(~coords.valid_mask))
        self.assertAlmostEqual(
            coords.polar[:, 0].max().item(), math.sqrt(2.0), places=5)

    def test_square_isotropic_masked_samples_are_spatial_padding_candidates(self):
        coords = SamplingCoords(
            16.0, 0.5, 16, style="isotropic", fov_type="square")

        self.assertGreater(len(coords.fov_padding_coords), 0)
        square_half_extent = 1.0
        self.assertTrue(torch.all(
            coords.fov_padding_coords.abs().amax(dim=1)
            > square_half_extent))

        layer = KNNGetterLayer(
            25, coords, coords, device="cpu", sample_cortex=False)
        first_fov_pad = len(coords)
        past_fov_pad = first_fov_pad + len(coords.fov_padding_coords)
        selected_fov_padding = torch.logical_and(
            layer.knn_indices >= first_fov_pad,
            layer.knn_indices < past_fov_pad)

        self.assertTrue(torch.any(selected_fov_padding))
        self.assertTrue(torch.all(
            layer.knn_indices_pad_mask[selected_fov_padding]))
        self.assertTrue(torch.all(
            layer.knn_indices_pad_token[selected_fov_padding] == len(coords)))


class TestWarpedCartesian(unittest.TestCase):
    def test_inverse_mapping_matches_warp_visualization(self):
        fov = 16.0
        cmf_a = 0.5
        cartesian, polar, plotting, valid = (
            get_warped_cartesian_sampling_coords(
                fov, cmf_a, 32, fov_type="circular",
                return_valid_mask=True))

        radius_deg = polar[:, 0] * (fov / 2.0)
        rho_axis = math.log((fov / 2.0 + cmf_a) / cmf_a)
        warped_radius = torch.log((radius_deg + cmf_a) / cmf_a) / rho_axis
        reconstructed = torch.stack((
            warped_radius * torch.cos(polar[:, 1]),
            warped_radius * torch.sin(polar[:, 1])), dim=1)

        torch.testing.assert_close(reconstructed, plotting, atol=2e-6, rtol=2e-6)
        expected_valid = torch.linalg.vector_norm(
            cartesian, dim=1) <= 1.0 + 1e-6
        torch.testing.assert_close(valid, expected_valid)

    def test_square_warp_uses_circular_map_and_outer_square_mask(self):
        circle = SamplingCoords(
            16.0, 0.5, 32, style="warped_cartesian", fov_type="circular")
        square = SamplingCoords(
            16.0, 0.5, 32, style="warped_cartesian", fov_type="square")
        square_half_extent = 1.0

        self.assertEqual(len(circle), 32 ** 2)
        torch.testing.assert_close(circle.plotting, square.plotting)
        torch.testing.assert_close(circle.cartesian, square.cartesian)
        expected = (
            square.cartesian.abs().amax(dim=1)
            <= square_half_extent + 2e-6)
        torch.testing.assert_close(square.valid_mask, expected)
        self.assertTrue(torch.any(~square.valid_mask))
        self.assertTrue(torch.all(
            square.cartesian_pad_coords.abs().amax(dim=1) > 1.0))
        self.assertLess(
            circle.valid_mask.sum().item(), square.valid_mask.sum().item())

    def test_wang_fov_maps_the_full_native_square(self):
        fov = 16.0
        cmf_a = 0.5
        resolution = 32
        coords = SamplingCoords(
            fov, cmf_a, resolution, style="warped_cartesian",
            fov_type="wang")

        self.assertEqual(len(coords), resolution ** 2)
        self.assertTrue(torch.all(coords.valid_mask))
        self.assertEqual(len(coords.fov_padding_coords), 0)

        radius_normalizer = _warped_cartesian_radius_normalizer(
            "wang", fov, cmf_a)
        native_corner_radius = (
            math.sqrt(2.0) * (1.0 - 1.0 / resolution))
        expected_max_radius = cmf_a * math.expm1(
            native_corner_radius / radius_normalizer
            * math.log((fov / 2.0 + cmf_a) / cmf_a)) / (fov / 2.0)
        self.assertAlmostEqual(
            coords.polar[:, 0].max().item(), expected_max_radius, places=5)

        visual_grid = coords.cartesian.reshape(resolution, resolution, 2)
        corner_radius = torch.linalg.vector_norm(visual_grid[0, 0]).item()
        axis_edge_radius = torch.linalg.vector_norm(
            visual_grid[0, resolution // 2]).item()
        self.assertGreater(corner_radius, axis_edge_radius)

    def test_wang_fov_contains_the_outer_square(self):
        fov = 16.0
        cmf_a = 0.5
        square_half_extent = 1.0
        radius_normalizer = _warped_cartesian_radius_normalizer(
            "wang", fov, cmf_a)

        side_center, _ = _inverse_warped_cartesian(
            torch.tensor([[1.0, 0.0]]), fov, cmf_a,
            radius_normalizer=radius_normalizer)
        self.assertAlmostEqual(
            torch.linalg.vector_norm(side_center).item(),
            square_half_extent, places=6)

        angles = torch.linspace(0.0, 2.0 * math.pi, 257)[:-1]
        directions = torch.stack((torch.cos(angles), torch.sin(angles)), 1)
        native_boundary = directions / directions.abs().amax(
            dim=1, keepdim=True)
        warped_boundary, _ = _inverse_warped_cartesian(
            native_boundary, fov, cmf_a,
            radius_normalizer=radius_normalizer)
        warped_radius = torch.linalg.vector_norm(warped_boundary, dim=1)
        square_radius = square_half_extent / directions.abs().amax(dim=1)
        self.assertTrue(torch.all(warped_radius >= square_radius - 1e-6))

    def test_wang_fov_rejected_by_other_sensors(self):
        for style in ("isotropic", "isotropic_fixn", "logpolar",
                      "logpolar_as_grid", "uniform", "uniform_as_grid"):
            with self.subTest(style=style):
                with self.assertRaisesRegex(
                        ValueError, "only supported by"):
                    SamplingCoords(
                        16.0, 0.5, 16, style=style,
                        fov_type="wang")

    def test_unknown_fov_type_is_rejected(self):
        for fov_type in ("triangle", "warped_cartesian"):
            with self.subTest(fov_type=fov_type):
                with self.assertRaisesRegex(
                        ValueError, "fov_type must be one of"):
                    SamplingCoords(
                        16.0, 0.5, 16, style="warped_cartesian",
                        fov_type=fov_type)

    def test_flat_and_grid_outputs_are_identical_and_invalid_samples_zero(self):
        image = torch.ones(2, 3, 128, 128)
        fixation = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        for fov_type in ("circular", "square", "wang"):
            kwargs = dict(
                resolution=16, start_res=128, fov=16.0, cmf_a=0.5,
                sampler="grid_nn", fixation_size=32, device="cpu",
                auto_match_cart_resources=True, fov_type=fov_type)
            flat = RetinalTransform(
                style="warped_cartesian", **kwargs).eval()
            grid = RetinalTransform(
                style="warped_cartesian_as_grid", **kwargs).eval()

            flat_output = flat(image, fixation, 32)
            grid_output = grid(image, fixation, 32)

            self.assertEqual(flat_output.shape, (2, 3, 16 ** 2))
            self.assertEqual(grid_output.shape, (2, 3, 16, 16))
            torch.testing.assert_close(flat_output, grid_output.flatten(2))
            torch.testing.assert_close(flat.valid_mask, grid.valid_mask)
            if fov_type == "square":
                self.assertTrue(torch.any(~flat.valid_mask))
            self.assertTrue(torch.all(
                flat_output[:, :, ~flat.valid_mask] == 0))
            self.assertTrue(torch.all(
                flat_output[:, :, flat.valid_mask] == 1))

    def test_knn_convolution_zeros_invalid_output_locations(self):
        coords = SamplingCoords(
            16.0, 0.5, 16, style="warped_cartesian",
            fov_type="circular")
        layer = KNNConvLayer(
            1, 1, 9, coords, coords, device="cpu", sample_cortex=True,
            kernel_backend="baseline", bias=True)
        with torch.no_grad():
            layer.weight.fill_(1.0)
            layer.bias.fill_(7.0)
        features = coords.valid_mask.to(torch.float32)[None, None, :]

        output = layer(features)

        self.assertTrue(torch.all(
            output[:, :, ~coords.valid_mask] == 0))

    def test_warped_cartesian_padding_covers_deep_corner_neighborhoods(self):
        coords = SamplingCoords(
            16.0, 0.5, 16, style="warped_cartesian", fov_type="wang")
        axis = torch.unique(coords.cortical[:, 0]).sort().values
        step = axis[1] - axis[0]
        expected_corner = torch.stack((
            axis[0] - 3 * step,
            axis[0] - 3 * step,
        ))

        self.assertTrue(torch.any(torch.all(torch.isclose(
            coords.cortical_pad_coords, expected_corner[None],
            atol=1e-6, rtol=1e-6), dim=1)))

    def test_warped_cartesian_accepts_geodesic_sampling_request(self):
        in_coords = SamplingCoords(
            16.0, 0.5, 8, style="warped_cartesian", fov_type="wang")
        out_coords = SamplingCoords(
            16.0, 0.5, 2, style="warped_cartesian", fov_type="wang")

        layer = KNNGetterLayer(
            4, in_coords, out_coords,
            device="cpu", sample_cortex="geodesic")

        self.assertEqual(layer.knn_indices.shape, (4, len(out_coords)))

    def test_logpolar_square_invalid_samples_use_zero_padding(self):
        retina = RetinalTransform(
            resolution=16, start_res=64, fov=16.0, cmf_a=0.5,
            style="logpolar_as_grid", sampler="grid_nn",
            fixation_size=32, device="cpu", fov_type="square").eval()
        output = retina(
            torch.ones(1, 3, 64, 64), torch.tensor([[0.5, 0.5]]), 32)

        flat = output.flatten(2)
        self.assertTrue(torch.all(flat[:, :, ~retina.valid_mask] == 0))
        self.assertTrue(torch.all(flat[:, :, retina.valid_mask] == 1))

    def test_square_grid_masked_cells_are_knn_padding(self):
        coords = SamplingCoords(
            16.0, 0.5, 16, style="logpolar", fov_type="square")
        layer = KNNGetterLayer(
            25, coords, coords, device="cpu", sample_cortex=False)

        selected_real = layer.knn_indices < len(coords)
        selected_masked = torch.zeros_like(selected_real)
        selected_masked[selected_real] = ~coords.valid_mask[
            layer.knn_indices[selected_real]]

        self.assertTrue(torch.any(selected_masked))
        self.assertTrue(torch.all(
            layer.knn_indices_pad_mask[selected_masked]))
        self.assertTrue(torch.all(
            layer.knn_indices_pad_token[selected_masked] == len(coords)))

        features = torch.ones(1, 1, len(coords))
        features[:, :, ~coords.valid_mask] = 123.0
        neighborhoods = layer(features)
        selected_masked_out = selected_masked.unsqueeze(0).unsqueeze(0)
        self.assertTrue(torch.all(torch.isnan(
            neighborhoods[selected_masked_out])))

    def test_logpolar_outer_radius_uses_spatial_padding(self):
        coords = SamplingCoords(
            16.0, 0.5, 16, style="logpolar", fov_type="circular")
        layer = KNNGetterLayer(
            49, coords, coords, device="cpu", sample_cortex=True)
        outer_index = torch.argmax(coords.polar[:, 0]).item()

        self.assertTrue(torch.any(
            layer.knn_indices_pad_mask[:, outer_index]))

    def test_grid_topologies_can_pool_to_one_cortical_location(self):
        for style in ("warped_cartesian", "logpolar"):
            with self.subTest(style=style):
                in_coords = SamplingCoords(
                    16.0, 0.5, 4, style=style, fov_type="square")
                out_coords = SamplingCoords(
                    16.0, 0.5, 1, style=style, fov_type="square")
                layer = KNNGetterLayer(
                    4, in_coords, out_coords,
                    device="cpu", sample_cortex=True)

                self.assertEqual(layer.knn_indices.shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
