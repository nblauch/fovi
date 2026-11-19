"""Modal pooling: each output takes the most common value among its pooling neighbors.

Used by ``mode="modal"`` in :class:`~fovi.arch.knn.KNNPoolingLayer` and
:class:`~fovi.arch.pooling.GridPoolingLayer`, for label maps such as semantic or
instance segmentation, which arithmetic pooling would mix into labels no object has.
"""

import torch

__all__ = ["modal_pool"]

# Upper bound on ``batch * outputs * k`` elements per chunk. Each chunk holds about
# ten int64 tensors of this size, so peak scratch memory stays near 300 MB.
_CHUNK_ELEMENTS = 2**22


def modal_pool(values: torch.Tensor, neighbors: torch.Tensor, visible: torch.Tensor | None = None) -> torch.Tensor:
    """Pool ``(B, C, N_in)`` values to ``(B, C, N_out)`` by majority vote over a neighbor table.

    Each output takes the most common full value (all ``C`` channels together)
    among its neighbors, so neighbors ``[0, 0, 0]``, ``[0, 0, 0]`` and
    ``[0, 0, 1]`` give ``[0, 0, 0]``. Values are compared bitwise and copied from
    an input, in its dtype, never converted or mixed: integer IDs of any width
    and colorized label vectors come back exactly.

    Ties go to the value whose first occurrence comes earliest in the table's
    neighbor order. Excluded neighbors (the table's pad token, ``N_in``) and,
    if given, inputs the image does not see do not vote. Outputs without a voting
    neighbor are zero. The vote runs on the device without a host
    synchronization, in chunks that bound its scratch memory.

    Args:
        values: ``(B, C, N_in)`` values of any dtype.
        neighbors: ``(k, N_out)`` input indices of each output's neighbors in
            tie-break order; ``N_in`` marks an excluded neighbor.
        visible: Optional ``(B, N_in)`` boolean mask of the inputs each image sees.

    Returns:
        ``(B, C, N_out)`` pooled values with the dtype of ``values``.
    """
    batch, channels, num_inputs = values.shape
    k, num_outputs = neighbors.shape
    if visible is not None and tuple(visible.shape) != (batch, num_inputs):
        raise ValueError(f"visible must have shape {(batch, num_inputs)}, got {tuple(visible.shape)}")
    # (B, N_in, C) with unit channel stride, which the byte view of the keys needs; with one
    # channel, ``contiguous()`` would keep the transposed view's stride.
    samples = torch.empty((batch, num_inputs, channels), dtype=values.dtype, device=values.device)
    samples.copy_(values.transpose(1, 2))
    # Equality keys, padded with one never-voting input at index N_in.
    keys = torch.nn.functional.pad(_equality_keys(samples), (0, 1))  # (B, N_in + 1)
    table = neighbors.T.to(device=values.device, dtype=torch.long)  # (N_out, k)
    valid = table != num_inputs  # (N_out, k)
    chunk = max(1, _CHUNK_ELEMENTS // max(1, num_outputs * k))
    seen = None if visible is None else torch.nn.functional.pad(visible.to(torch.bool), (0, 1)).split(chunk)
    slots = []
    for index, chunk_keys in enumerate(keys.split(chunk)):
        chunk_valid = valid if seen is None else valid & seen[index][:, table]  # (N_out, k) or (b, N_out, k)
        slots.append(_modal_slot(chunk_keys[:, table], chunk_valid))
    slot = torch.cat(slots)  # (B, N_out); k where no neighbor votes
    has_value = slot < k
    # Input index per output; outputs without a value read input 0 and are zeroed below.
    source = table.gather(1, slot.clamp(max=k - 1).T).T.clamp(max=num_inputs - 1)
    pooled = samples.gather(1, source.unsqueeze(-1).expand(-1, -1, channels))
    pooled = torch.where(has_value.unsqueeze(-1), pooled, torch.zeros((), dtype=values.dtype, device=values.device))
    return pooled.transpose(1, 2)


def _equality_keys(values: torch.Tensor) -> torch.Tensor:
    """Map ``(B, N, C)`` channel vectors to ``(B, N)`` int64 keys that are equal iff the vectors are bitwise equal.

    Vectors of at most 8 bytes are packed into the key directly; wider vectors
    fall back to dense ranks from :func:`torch.unique`.
    """
    batch, num, _ = values.shape
    raw = values.view(torch.uint8) if values.dtype != torch.bool else values.to(torch.uint8)
    num_bytes = raw.shape[-1]
    if num_bytes <= 8:
        return torch.nn.functional.pad(raw, (0, 8 - num_bytes)).view(torch.int64).reshape(batch, num)
    return torch.unique(raw.reshape(batch * num, num_bytes), dim=0, return_inverse=True)[1].reshape(batch, num)


def _modal_slot(keys: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Pick the modal neighbor slot of ``(B, N_out, k)`` keys; ``valid`` is the ``(N_out, k)`` or ``(B, N_out, k)``
    voting mask.

    Returns the ``(B, N_out)`` earliest slot holding the most common valid key,
    or ``k`` where no slot is valid.
    """
    k = keys.shape[-1]
    sorted_keys, order = torch.sort(keys, dim=-1)
    # Label equal-key runs in sorted order, then count valid votes and find the earliest valid slot per run.
    new_run = torch.ones_like(sorted_keys, dtype=torch.bool)
    new_run[..., 1:] = sorted_keys[..., 1:] != sorted_keys[..., :-1]
    run = new_run.cumsum(-1) - 1
    valid_sorted = valid.expand_as(order).gather(-1, order)
    votes = torch.zeros_like(run).scatter_add_(-1, run, valid_sorted.long())
    earliest = torch.full_like(run, k).scatter_reduce_(-1, run, torch.where(valid_sorted, order, k), "amin")
    # Most votes first, then earliest slot; runs without valid votes score zero.
    best = (votes * (k + 1) + (k - earliest)).argmax(-1, keepdim=True)
    return earliest.gather(-1, best).squeeze(-1)
