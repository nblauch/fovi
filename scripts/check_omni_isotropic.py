"""Exercise the real pretrained omni model, optimizer, and checkpoint on one GPU."""

import argparse
import json
from pathlib import Path

import torch

from fovi.models.fovinet import FoviNet
from fovi.models.loading import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--config-name", default="dinov3_isotropic_multi_cmf")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(1)
    torch.set_num_threads(1)
    config, _, _ = load_config(
        args.config_name,
        False,
        Path(__file__).resolve().parents[1] / "config",
        device="cuda",
    )
    model = FoviNet(config, device="cuda").train()
    assert model.retinal_transform.sampler.coords.cartesian.shape == (7, 16384, 2)
    patch = model.network.backbone.embeddings.patch_embeddings
    assert patch.num_patches == 64
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.base_lr,
        eps=config.training.eps,
        weight_decay=config.training.weight_decay,
    )
    images = torch.rand(args.batch_size, 3, 256, 256, device="cuda")
    levels = torch.arange(args.batch_size, device="cuda") % 7
    labels = torch.arange(args.batch_size, device="cuda") % 1000
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images, cmf_indices=levels.roll(step))[0]
            loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if not all(g is not None and torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("Missing or nonfinite training gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"Smoke optimizer step {step}: loss={losses[-1]:.6f}", flush=True)
    model.eval()
    fixations = [torch.full((args.batch_size, 2), 0.5, device="cuda")]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(
            images, cmf_indices=levels, fixations=fixations, n_fixations=1
        )[0]
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "smoke.pth"
    torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model(images, cmf_indices=levels, fixations=fixations, n_fixations=1)[
            0
        ]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    report = {
        "config_name": args.config_name,
        "position_coordinate_space": config.model.vit.position_coordinate_space,
        "sample_cortex": config.saccades.sample_cortex,
        "losses": losses,
        "retinal_samples": 16384,
        "patch_tokens": 64,
        "reference_side": patch.reference_side,
        "patch_max_coord_val": patch.patch_max_coord_val,
        "max_patch_position_norm_per_level": patch.positions.norm(dim=-1)
        .amax(dim=-1)
        .tolist(),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "levels": list(config.saccades.cmf_a),
        "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__,
        "checkpoint_roundtrip": True,
    }
    (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
