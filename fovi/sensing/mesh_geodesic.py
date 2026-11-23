"""Exact geodesic distances on triangle meshes, solved in batches with Warp.

Distances are measured on the piecewise-linear surface itself, not along mesh
edges. The solver follows the window-propagation scheme of Chen and Han (1990)
with the vertex-distance filtering and priority ordering of Xin and Wang's
"Improved Chen-Han" algorithm (2009), the same exact problem that
Mitchell-Mount-Papadimitriou solvers such as ``pygeodesic`` answer.

A *window* is an interval of a mesh edge whose shortest paths all unfold to
straight lines from one (pseudo-)source. Windows cross faces in order of their
nearest distance, and each crossing unfolds the next face into the plane of the
current one. Paths that bend around a saddle or boundary vertex continue from
that vertex as a new pseudo-source. A window is discarded only when one of its
edge's endpoint vertices already reaches every point of the window by a strictly
shorter path, so no shortest path is lost.

Each single-source solve is one Warp thread, in float64, with its own distance
array and window heap. A batch of solves, such as one per source on a shared
mesh or one per local mesh of a KNN layer, runs as one CUDA launch, or as
concurrent launches on Warp's CPU backend, one per core. Each solve is serial,
so CUDA pays off for many small meshes; a few solves on a large mesh run faster
on the CPU.
"""

# No ``from __future__ import annotations``: Warp evaluates the kernel
# annotations, which refer to names local to ``_solve_kernel``.
import os
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

__all__ = ["ExactGeodesicSolver", "SOLVER_VERSION", "distances_from_first_vertex"]

SOLVER_VERSION = "fovi-ich-1"
"""Identifies this solver in persistent caches of its results."""

# A vertex whose face angles sum past 2*pi by more than this is a saddle.
_SADDLE_ANGLE_TOLERANCE = 1e-9
# Relative tolerance on edge length for interval and vertex coincidence tests.
_EDGE_TOLERANCE = 1e-10
# Unreached vertices hold this value inside the kernel and inf in results.
_UNREACHED = 1e300
# Initial window-heap capacity per solve, as a multiple of its vertex count;
# solves that overflow it rerun with four times the capacity.
_HEAP_PER_VERTEX = 4
# Upper bound on concurrent CPU launches.
_MAX_CPU_THREADS = 32
_FIRST_LAUNCH_LOCK = threading.Lock()
_LAUNCHED_DEVICES: set = set()


@dataclass
class _Mesh:
    """Half-edge arrays of a mesh, possibly a disjoint union of several meshes.

    Half-edge ``3 * f + j`` is face ``f``'s edge from corner ``j`` to corner
    ``j + 1``, with corner ``j + 2`` opposite. Its frame puts the lower vertex
    index at the origin, the higher at ``(length, 0)``, and the opposite vertex at
    ``(cx, cy)`` with ``cy > 0``. Windows on a half-edge propagate into its face.
    """

    num_vertices: int
    start: np.ndarray
    end: np.ndarray
    third: np.ndarray
    length: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    # Per half-edge, its face's other edges A-C (slot 0) and C-B (slot 1): the
    # half-edge beyond each (-1 on the boundary), whether the child frame runs
    # from C, and the rigid map into it (origin, x axis, y axis, child length).
    child: np.ndarray  # (2H,) int
    child_flip: np.ndarray  # (2H,) int
    child_frame: np.ndarray  # (2H, 7) float64: ox, oy, ux, uy, nx, ny, length
    pseudo_source: np.ndarray  # (V,) int: saddle or boundary vertex
    # CSR per vertex: windows it emits as a source, and its edge neighbors.
    emit_offset: np.ndarray
    emit_half: np.ndarray
    emit_position: np.ndarray  # (E, 2) the vertex in the emitted window's frame
    neighbor_offset: np.ndarray
    neighbor: np.ndarray
    neighbor_length: np.ndarray


def _validated(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise ValueError(f"vertices must have shape (V, 2) or (V, 3), got {vertices.shape}")
    if vertices.shape[1] == 2:
        vertices = np.pad(vertices, ((0, 0), (0, 1)))
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("faces reference vertices outside the vertex array")
    return vertices, faces


def _csr(groups: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable order grouping items by ``groups`` and the per-group offsets."""
    order = np.argsort(groups, kind="stable")
    offset = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(np.bincount(groups, minlength=count), out=offset[1:])
    return order, offset


def _prepare(vertices: np.ndarray, faces: np.ndarray) -> _Mesh:
    num_vertices = len(vertices)
    if ((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
            | (faces[:, 0] == faces[:, 2])).any():
        raise ValueError("a face repeats a vertex")
    p = faces
    q = np.roll(faces, -1, axis=1)
    r = np.roll(faces, -2, axis=1)
    start = np.minimum(p, q).reshape(-1)
    end = np.maximum(p, q).reshape(-1)
    third = r.reshape(-1)
    length = np.linalg.norm(vertices[end] - vertices[start], axis=1)
    to_start = np.linalg.norm(vertices[third] - vertices[start], axis=1)
    to_end = np.linalg.norm(vertices[third] - vertices[end], axis=1)
    if not (length > 0).all():
        raise ValueError("the mesh has a zero-length edge")
    cx = (length**2 + to_start**2 - to_end**2) / (2 * length)
    # Triangulating in one chart and measuring in another can flatten a face to
    # zero area, such as a hull sliver whose corners are collinear on the
    # cortex. Give such faces a minimal height, which keeps them crossable and
    # changes distances by at most that height.
    min_height = _EDGE_TOLERANCE * np.maximum(length, np.maximum(to_start, to_end))
    cy = np.maximum(np.sqrt(np.maximum(to_start**2 - cx**2, 0.0)), min_height)

    # Twins: the other half-edge on the same edge.
    num_half = len(start)
    key = start * np.int64(num_vertices) + end
    order = np.argsort(key, kind="stable")
    _, first, counts = np.unique(key[order], return_index=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError("an edge borders more than two faces; the mesh must be edge-manifold")
    twin = np.full(num_half, -1, dtype=np.int64)
    pairs = first[counts == 2]
    twin[order[pairs]], twin[order[pairs + 1]] = order[pairs + 1], order[pairs]

    half = np.arange(num_half)
    face_base = 3 * (half // 3)
    following = face_base + (half + 1) % 3  # edge (corner j+1, corner j+2)
    preceding = face_base + (half + 2) % 3  # edge (corner j+2, corner j)
    a_is_corner_j = p.reshape(-1) == start
    edge_ac = np.where(a_is_corner_j, preceding, following)
    edge_cb = np.where(a_is_corner_j, following, preceding)
    zeros = np.zeros(num_half)
    position = {"A": (zeros, zeros), "B": (length, zeros), "C": (cx, cy)}

    def slot(edge, first_name, second_name, remaining_name):
        first_vertex = start if first_name == "A" else third
        second_vertex = end if second_name == "B" else third
        flip = second_vertex < first_vertex  # child frames start at the lower index
        fx, fy = position[first_name]
        gx, gy = position[second_name]
        ox, oy = np.where(flip, gx, fx), np.where(flip, gy, fy)
        ex, ey = np.where(flip, fx, gx), np.where(flip, fy, gy)
        child_length = np.hypot(ex - ox, ey - oy)
        ux, uy = (ex - ox) / child_length, (ey - oy) / child_length
        nx, ny = -uy, ux
        rx, ry = position[remaining_name]
        away = (rx - ox) * nx + (ry - oy) * ny > 0  # n must point out of this face
        nx, ny = np.where(away, -nx, nx), np.where(away, -ny, ny)
        frame = np.stack((ox, oy, ux, uy, nx, ny, child_length), axis=1)
        return twin[edge], flip, frame

    left = slot(edge_ac, "A", "C", "B")
    right = slot(edge_cb, "C", "B", "A")
    child = np.stack((left[0], right[0]), axis=1).reshape(-1)
    child_flip = np.stack((left[1], right[1]), axis=1).reshape(-1)
    child_frame = np.stack((left[2], right[2]), axis=1).reshape(-1, 7)

    # Angle sums and boundary flags decide which vertices act as pseudo-sources.
    u = vertices[q] - vertices[p]
    w = vertices[r] - vertices[p]
    cos = np.einsum("fcd,fcd->fc", u, w) / (np.linalg.norm(u, axis=2) * np.linalg.norm(w, axis=2))
    angle_sum = np.zeros(num_vertices)
    np.add.at(angle_sum, p.reshape(-1), np.arccos(np.clip(cos, -1.0, 1.0)).reshape(-1))
    boundary = np.zeros(num_vertices, dtype=bool)
    open_edges = twin < 0
    boundary[start[open_edges]] = True
    boundary[end[open_edges]] = True
    referenced = np.zeros(num_vertices, dtype=bool)
    referenced[faces.reshape(-1)] = True
    saddle = angle_sum > 2 * np.pi + _SADDLE_ANGLE_TOLERANCE
    pseudo_source = referenced & (boundary | saddle)

    # A vertex emits windows across the opposite edge of each incident face,
    # into the face beyond, where it sits at (cx, -cy).
    inner = np.nonzero(twin >= 0)[0]
    emit_order, emit_offset = _csr(third[inner], num_vertices)
    inner = inner[emit_order]
    # Each edge once (its first half-edge in sorted order), in both directions.
    unique_half = order[first]
    tails = np.concatenate((start[unique_half], end[unique_half]))
    heads = np.concatenate((end[unique_half], start[unique_half]))
    lengths = np.concatenate((length[unique_half], length[unique_half]))
    neighbor_order, neighbor_offset = _csr(tails, num_vertices)

    return _Mesh(
        num_vertices=num_vertices, start=start, end=end, third=third,
        length=length, cx=cx, cy=cy, child=child, child_flip=child_flip.astype(np.int64),
        child_frame=child_frame, pseudo_source=pseudo_source.astype(np.int64),
        emit_offset=emit_offset, emit_half=twin[inner],
        emit_position=np.stack((cx[inner], -cy[inner]), axis=1),
        neighbor_offset=neighbor_offset, neighbor=heads[neighbor_order],
        neighbor_length=lengths[neighbor_order])


@cache
def _solve_kernel():
    import warp as wp

    wp.config.quiet = True
    wp.init()
    f64 = wp.float64
    int_array = wp.array(dtype=wp.int32)
    float_array = wp.array(dtype=wp.float64)

    @wp.func
    def heap_push(key: f64, code: int, sx: f64, sy: f64, sigma: f64, b0: f64, b1: f64,
                  b: int, base: int, capacity: int, heap_size: int_array, heap_key: float_array,
                  heap_code: int_array, heap_value: float_array, status: int_array):
        n = heap_size[b]
        if n >= capacity:
            status[b] = 1
        else:
            i = int(n)
            while i > 0:
                parent = (i - 1) // 2
                if heap_key[base + parent] <= key:
                    break
                heap_key[base + i] = heap_key[base + parent]
                heap_code[base + i] = heap_code[base + parent]
                for field in range(5):
                    heap_value[(base + i) * 5 + field] = heap_value[(base + parent) * 5 + field]
                i = parent
            heap_key[base + i] = key
            heap_code[base + i] = code
            heap_value[(base + i) * 5 + 0] = sx
            heap_value[(base + i) * 5 + 1] = sy
            heap_value[(base + i) * 5 + 2] = sigma
            heap_value[(base + i) * 5 + 3] = b0
            heap_value[(base + i) * 5 + 4] = b1
            heap_size[b] = n + 1

    @wp.func
    def heap_pop(b: int, base: int, heap_size: int_array, heap_key: float_array,
                 heap_code: int_array, heap_value: float_array):
        n = heap_size[b] - 1
        heap_size[b] = n
        if n > 0:
            key = heap_key[base + n]
            code = heap_code[base + n]
            v0 = heap_value[(base + n) * 5 + 0]
            v1 = heap_value[(base + n) * 5 + 1]
            v2 = heap_value[(base + n) * 5 + 2]
            v3 = heap_value[(base + n) * 5 + 3]
            v4 = heap_value[(base + n) * 5 + 4]
            i = int(0)
            while 2 * i + 1 < n:
                c = int(2 * i + 1)
                if c + 1 < n:
                    if heap_key[base + c + 1] < heap_key[base + c]:
                        c = c + 1
                if heap_key[base + c] >= key:
                    break
                heap_key[base + i] = heap_key[base + c]
                heap_code[base + i] = heap_code[base + c]
                for field in range(5):
                    heap_value[(base + i) * 5 + field] = heap_value[(base + c) * 5 + field]
                i = c
            heap_key[base + i] = key
            heap_code[base + i] = code
            heap_value[(base + i) * 5 + 0] = v0
            heap_value[(base + i) * 5 + 1] = v1
            heap_value[(base + i) * 5 + 2] = v2
            heap_value[(base + i) * 5 + 3] = v3
            heap_value[(base + i) * 5 + 4] = v4

    @wp.func
    def reach(v: int, d: f64, b: int, base: int, capacity: int, dist_base: int, vertex_base: int,
              dist: float_array, pseudo: int_array, heap_size: int_array, heap_key: float_array,
              heap_code: int_array, heap_value: float_array, status: int_array):
        k = dist_base + v - vertex_base
        if d < dist[k]:
            dist[k] = d
            if pseudo[v] != 0:
                zero = f64(0.0)
                heap_push(d, -1 - v, zero, zero, d, zero, zero, b, base, capacity,
                          heap_size, heap_key, heap_code, heap_value, status)

    @wp.func
    def dominated(h: int, sx: f64, sy: f64, sigma: f64, b0: f64, b1: f64, dist_base: int,
                  vertex_base: int, dist: float_array, start: int_array, end: int_array,
                  length: float_array) -> int:
        # Along the edge, the margin of a window over an endpoint-vertex path is
        # monotone, so testing the far end of the interval covers the window.
        L = length[h]
        margin = f64(1.0e-12) * (sigma + L)
        result = int(0)
        da = dist[dist_base + start[h] - vertex_base]
        if da + b1 < sigma + wp.sqrt((b1 - sx) * (b1 - sx) + sy * sy) - margin:
            result = 1
        db = dist[dist_base + end[h] - vertex_base]
        if db + (L - b0) < sigma + wp.sqrt((b0 - sx) * (b0 - sx) + sy * sy) - margin:
            result = 1
        return result

    @wp.func
    def push_window(h: int, sx: f64, sy: f64, sigma: f64, lo: f64, hi: f64,
                    b: int, base: int, capacity: int, dist_base: int, vertex_base: int,
                    dist: float_array, pseudo: int_array, start: int_array, end: int_array,
                    length: float_array, heap_size: int_array, heap_key: float_array,
                    heap_code: int_array, heap_value: float_array, status: int_array):
        L = length[h]
        tol = f64(1.0e-10) * L
        b0 = wp.max(lo, f64(0.0))
        b1 = wp.min(hi, L)
        if b1 - b0 > tol:
            if b0 <= tol:
                b0 = f64(0.0)
                reach(start[h], sigma + wp.sqrt(sx * sx + sy * sy), b, base, capacity,
                      dist_base, vertex_base, dist, pseudo, heap_size, heap_key, heap_code,
                      heap_value, status)
            if b1 >= L - tol:
                b1 = L
                reach(end[h], sigma + wp.sqrt((L - sx) * (L - sx) + sy * sy), b, base, capacity,
                      dist_base, vertex_base, dist, pseudo, heap_size, heap_key, heap_code,
                      heap_value, status)
            if dominated(h, sx, sy, sigma, b0, b1, dist_base, vertex_base, dist, start, end,
                         length) == 0:
                key = sigma - sy
                if sx < b0:
                    key = sigma + wp.sqrt((b0 - sx) * (b0 - sx) + sy * sy)
                if sx > b1:
                    key = sigma + wp.sqrt((sx - b1) * (sx - b1) + sy * sy)
                heap_push(key, h, sx, sy, sigma, b0, b1, b, base, capacity,
                          heap_size, heap_key, heap_code, heap_value, status)

    @wp.func
    def emit_child(k: int, sx: f64, sy: f64, sigma: f64, t0: f64, t1: f64,
                   child: int_array, child_flip: int_array, child_frame: float_array,
                   b: int, base: int, capacity: int, dist_base: int, vertex_base: int,
                   dist: float_array, pseudo: int_array, start: int_array, end: int_array,
                   length: float_array, heap_size: int_array, heap_key: float_array,
                   heap_code: int_array, heap_value: float_array, status: int_array):
        target = child[k]
        if target >= 0:
            dx = sx - child_frame[k * 7 + 0]
            dy = sy - child_frame[k * 7 + 1]
            csx = dx * child_frame[k * 7 + 2] + dy * child_frame[k * 7 + 3]
            csy = dx * child_frame[k * 7 + 4] + dy * child_frame[k * 7 + 5]
            # A source on the child edge's line only grazes it.
            if csy < f64(0.0):
                lo = t0
                hi = t1
                if child_flip[k] != 0:
                    lo = f64(1.0) - t1
                    hi = f64(1.0) - t0
                child_length = child_frame[k * 7 + 6]
                push_window(target, csx, csy, sigma, lo * child_length, hi * child_length,
                            b, base, capacity, dist_base, vertex_base, dist, pseudo, start, end,
                            length, heap_size, heap_key, heap_code, heap_value, status)

    @wp.func
    def ray_t(sx: f64, sy: f64, x: f64, px: f64, py: f64, qx: f64, qy: f64) -> f64:
        # Parameter along p->q where the ray from s through (x, 0) crosses it.
        dx = x - sx
        dy = -sy
        ex = qx - px
        ey = qy - py
        t = ((sx - px) * dy - (sy - py) * dx) / (ex * dy - ey * dx)
        return wp.clamp(t, f64(0.0), f64(1.0))

    @wp.kernel(module="unique", enable_backward=False)
    def solve(source: int_array, vertex_start: int_array, vertex_count: int_array,
              dist_start: int_array, heap_start: int_array, heap_capacity: int_array,
              start: int_array, end: int_array, third: int_array, length: float_array,
              cx: float_array, cy: float_array, child: int_array, child_flip: int_array,
              child_frame: float_array, pseudo: int_array, emit_offset: int_array,
              emit_half: int_array, emit_position: float_array, neighbor_offset: int_array,
              neighbor: int_array, neighbor_length: float_array, dist: float_array,
              heap_size: int_array, heap_key: float_array, heap_code: int_array,
              heap_value: float_array, status: int_array):
        b = wp.tid()
        s = source[b]
        vertex_base = vertex_start[b]
        dist_base = dist_start[b]
        base = heap_start[b]
        capacity = heap_capacity[b]
        for i in range(vertex_count[b]):
            dist[dist_base + i] = f64(1.0e300)
        heap_size[b] = 0
        status[b] = 0
        zero = f64(0.0)
        reach(s, zero, b, base, capacity, dist_base, vertex_base, dist, pseudo,
              heap_size, heap_key, heap_code, heap_value, status)
        if pseudo[s] == 0:
            heap_push(zero, -1 - s, zero, zero, zero, zero, zero, b, base, capacity,
                      heap_size, heap_key, heap_code, heap_value, status)

        while heap_size[b] > 0 and status[b] == 0:
            code = heap_code[base]
            sx = heap_value[base * 5 + 0]
            sy = heap_value[base * 5 + 1]
            sigma = heap_value[base * 5 + 2]
            b0 = heap_value[base * 5 + 3]
            b1 = heap_value[base * 5 + 4]
            heap_pop(b, base, heap_size, heap_key, heap_code, heap_value)
            if code < 0:
                # A pseudo-source vertex, unless a shorter path to it superseded this event.
                v = -1 - code
                if sigma <= dist[dist_base + v - vertex_base]:
                    for e in range(neighbor_offset[v], neighbor_offset[v + 1]):
                        reach(neighbor[e], sigma + neighbor_length[e], b, base, capacity,
                              dist_base, vertex_base, dist, pseudo, heap_size, heap_key,
                              heap_code, heap_value, status)
                    for e in range(emit_offset[v], emit_offset[v + 1]):
                        h_emit = emit_half[e]
                        push_window(h_emit, emit_position[2 * e], emit_position[2 * e + 1], sigma,
                                    zero, length[h_emit], b, base, capacity, dist_base, vertex_base,
                                    dist, pseudo, start, end, length, heap_size, heap_key,
                                    heap_code, heap_value, status)
            else:
                h = code
                if dominated(h, sx, sy, sigma, b0, b1, dist_base, vertex_base, dist, start, end,
                             length) == 0:
                    L = length[h]
                    tol = f64(1.0e-10) * L
                    ccx = cx[h]
                    ccy = cy[h]
                    # Where the ray from the source through the opposite vertex C meets the edge.
                    xc = sx + (ccx - sx) * (-sy) / (ccy - sy)
                    if xc >= b0 - tol and xc <= b1 + tol:
                        reach(third[h], sigma + wp.sqrt((ccx - sx) * (ccx - sx) + (ccy - sy) * (ccy - sy)),
                              b, base, capacity, dist_base, vertex_base, dist, pseudo, heap_size,
                              heap_key, heap_code, heap_value, status)
                    if b0 < xc - tol:  # part of the window sees edge A-C
                        t0 = f64(0.0)
                        if b0 > zero:
                            t0 = ray_t(sx, sy, b0, zero, zero, ccx, ccy)
                        t1 = f64(1.0)
                        if b1 < xc:
                            t1 = ray_t(sx, sy, b1, zero, zero, ccx, ccy)
                        emit_child(2 * h, sx, sy, sigma, t0, t1, child, child_flip, child_frame,
                                   b, base, capacity, dist_base, vertex_base, dist, pseudo, start,
                                   end, length, heap_size, heap_key, heap_code, heap_value, status)
                    if b1 > xc + tol:  # part of the window sees edge C-B
                        t0 = f64(0.0)
                        if b0 > xc:
                            t0 = ray_t(sx, sy, b0, ccx, ccy, L, zero)
                        t1 = f64(1.0)
                        if b1 < L:
                            t1 = ray_t(sx, sy, b1, ccx, ccy, L, zero)
                        emit_child(2 * h + 1, sx, sy, sigma, t0, t1, child, child_flip,
                                   child_frame, b, base, capacity, dist_base, vertex_base, dist,
                                   pseudo, start, end, length, heap_size, heap_key, heap_code,
                                   heap_value, status)

    return solve


def _warp_device(device):
    import warp as wp

    wp.init()
    name = "cpu" if device is None else str(torch.device(device))
    if name.startswith("cuda") and not wp.is_cuda_available():
        name = "cpu"
    if name == "cuda":
        name = "cuda:0"
    return wp.get_device(name)


def _exclusive_cumsum(values: np.ndarray) -> np.ndarray:
    starts = np.zeros(len(values), dtype=np.int64)
    np.cumsum(values[:-1], out=starts[1:])
    return starts


def _solve(mesh: _Mesh, sources: np.ndarray, vertex_start: np.ndarray, vertex_count: np.ndarray,
           device) -> np.ndarray:
    """Distances for each source over its component's vertices, concatenated.

    CUDA solves the whole batch in one launch. Warp's CPU backend runs a launch
    on one core, so CPU batches are split across threads, whose launches run
    concurrently.
    """
    import warp as wp

    kernel = _solve_kernel()
    device = _warp_device(device)

    def int_array(values):
        return wp.array(np.ascontiguousarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    def float_array(values):
        return wp.array(np.ascontiguousarray(values, dtype=np.float64).reshape(-1),
                        dtype=wp.float64, device=device)

    mesh_arrays = [
        int_array(mesh.start), int_array(mesh.end), int_array(mesh.third),
        float_array(mesh.length), float_array(mesh.cx), float_array(mesh.cy),
        int_array(mesh.child), int_array(mesh.child_flip), float_array(mesh.child_frame),
        int_array(mesh.pseudo_source), int_array(mesh.emit_offset), int_array(mesh.emit_half),
        float_array(mesh.emit_position), int_array(mesh.neighbor_offset),
        int_array(mesh.neighbor), float_array(mesh.neighbor_length),
    ]

    def launch(problems: np.ndarray, capacity: np.ndarray):
        """Solve ``problems``; return their concatenated distances and overflow flags."""
        count = vertex_count[problems]
        total_heap = int(capacity.sum())
        if 5 * total_heap >= 2**31 or int(count.sum()) >= 2**31:
            raise MemoryError("Geodesic batch is too large for one launch")
        dist = wp.empty(int(count.sum()), dtype=wp.float64, device=device)
        heap_size = wp.zeros(len(problems), dtype=wp.int32, device=device)
        heap_key = wp.empty(total_heap, dtype=wp.float64, device=device)
        heap_code = wp.empty(total_heap, dtype=wp.int32, device=device)
        heap_value = wp.empty(5 * total_heap, dtype=wp.float64, device=device)
        status = wp.zeros(len(problems), dtype=wp.int32, device=device)
        inputs = [int_array(sources[problems]), int_array(vertex_start[problems]),
                  int_array(count), int_array(_exclusive_cumsum(count)),
                  int_array(_exclusive_cumsum(capacity)), int_array(capacity), *mesh_arrays,
                  dist, heap_size, heap_key, heap_code, heap_value, status]
        launched = device.alias in _LAUNCHED_DEVICES
        if not launched:
            # The first launch on a device compiles or loads the kernel; run it
            # alone so concurrent CPU worker threads do not each compile it.
            with _FIRST_LAUNCH_LOCK:
                if device.alias not in _LAUNCHED_DEVICES:
                    wp.launch(kernel, dim=len(problems), device=device, inputs=inputs)
                    _LAUNCHED_DEVICES.add(device.alias)
                else:
                    launched = True
        if launched:
            wp.launch(kernel, dim=len(problems), device=device, inputs=inputs)
        return dist.numpy(), status.numpy() != 0

    def solve_chunks(problems: np.ndarray, capacity: np.ndarray):
        if device.is_cuda or len(problems) == 1:
            return launch(problems, capacity)
        # Balance the chunks by vertex count, a proxy for each solve's work.
        workers = min(len(problems), os.cpu_count() or 1, _MAX_CPU_THREADS)
        weight = np.cumsum(vertex_count[problems])
        bounds = np.searchsorted(weight, weight[-1] * np.arange(1, workers) / workers)
        chunks = [chunk for chunk in np.split(np.arange(len(problems)), bounds) if len(chunk)]
        with ThreadPoolExecutor(len(chunks)) as pool:
            results = list(pool.map(lambda c: launch(problems[c], capacity[c]), chunks))
        return (np.concatenate([values for values, _ in results]),
                np.concatenate([overflow for _, overflow in results]))

    dist_start = _exclusive_cumsum(vertex_count)
    result = np.empty(int(vertex_count.sum()), dtype=np.float64)
    capacity = _HEAP_PER_VERTEX * vertex_count + 64
    pending = np.arange(len(sources))
    while len(pending):
        values, overflow = solve_chunks(pending, capacity[pending])
        local_start = _exclusive_cumsum(vertex_count[pending])
        for local in np.nonzero(~overflow)[0]:
            problem = pending[local]
            result[dist_start[problem]:dist_start[problem] + vertex_count[problem]] = (
                values[local_start[local]:local_start[local] + vertex_count[problem]])
        pending = pending[overflow]
        capacity[pending] *= 4
    result[result >= _UNREACHED] = np.inf
    return result


class ExactGeodesicSolver:
    """Exact surface geodesic distances from vertices of one triangle mesh.

    Args:
        vertices: ``(V, 3)`` or ``(V, 2)`` vertex positions; planar input lies
            in ``z = 0``.
        faces: ``(F, 3)`` vertex indices of the triangles. The mesh must be an
            edge-manifold (each edge borders at most two faces) of
            non-degenerate triangles. Vertices not referenced by any face are
            unreachable.
        device: Warp device for the solves; CPU by default, which suits a few
            solves on a large mesh.
    """

    def __init__(self, vertices, faces, device=None):
        vertices, faces = _validated(vertices, faces)
        self.num_vertices = len(vertices)
        self.device = device
        self._mesh = _prepare(vertices, faces)

    def distances(self, source: int) -> np.ndarray:
        """``(V,)`` float64 distances from one source vertex; ``inf`` if unreachable."""
        return self.distances_from_each([source])[0]

    def distances_from_each(self, sources: Iterable[int]) -> np.ndarray:
        """``(S, V)`` float64 distances, one row per source vertex, solved in one batch."""
        sources = np.asarray(list(sources), dtype=np.int64).reshape(-1)
        if sources.size and (sources.min() < 0 or sources.max() >= self.num_vertices):
            raise ValueError("source vertex index out of range")
        if not sources.size:
            return np.empty((0, self.num_vertices))
        count = np.full(len(sources), self.num_vertices, dtype=np.int64)
        flat = _solve(self._mesh, sources, np.zeros(len(sources), dtype=np.int64), count,
                      self.device)
        return flat.reshape(len(sources), self.num_vertices)


def distances_from_first_vertex(meshes: Sequence[tuple[np.ndarray, np.ndarray]],
                                device=None) -> list[np.ndarray]:
    """Distances from vertex 0 of each of many small meshes, solved in one batch.

    Args:
        meshes: ``(vertices, faces)`` pairs, as for :class:`ExactGeodesicSolver`.
        device: Warp device; CPU by default.

    Returns:
        One ``(V_i,)`` float64 array per mesh.
    """
    if not meshes:
        return []
    validated = [_validated(vertices, faces) for vertices, faces in meshes]
    counts = np.array([len(vertices) for vertices, _ in validated], dtype=np.int64)
    offsets = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    vertices = np.concatenate([vertices for vertices, _ in validated])
    faces = np.concatenate([faces + offset for (_, faces), offset in zip(validated, offsets)])
    flat = _solve(_prepare(vertices, faces), offsets, offsets, counts, device)
    return np.split(flat, offsets[1:])
