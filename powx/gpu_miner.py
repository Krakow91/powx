from __future__ import annotations

import time
from typing import Any, Callable

try:
    import numpy as np
    import pyopencl as cl
except Exception:  # pragma: no cover - optional dependency
    np = None
    cl = None


KERNEL_SOURCE = r"""
inline ulong rotl64(ulong x, uint r) {
    r &= 63u;
    return (x << r) | (x >> (64u - r));
}

inline int lt256(ulong a0, ulong a1, ulong a2, ulong a3, __global const ulong *target) {
    if (a0 < target[0]) return 1;
    if (a0 > target[0]) return 0;
    if (a1 < target[1]) return 1;
    if (a1 > target[1]) return 0;
    if (a2 < target[2]) return 1;
    if (a2 > target[2]) return 0;
    return a3 < target[3];
}

inline void kkhash(ulong nonce, __global const ulong *seed, __private ulong *h0, __private ulong *h1, __private ulong *h2, __private ulong *h3) {
    const ulong MASK = 0xFFFFFFFFFFFFFFFFUL;
    const ulong CONST_A = 0x9E3779B185EBCA87UL;
    const ulong CONST_B = 0xC2B2AE3D27D4EB4FUL;
    const ulong CONST_C = 0x165667B19E3779F9UL;
    const ulong CONST_D = 0x85EBCA77C2B2AE63UL;

    ulong x = (nonce ^ seed[4]) & MASK;
    ulong y = (seed[5] ^ CONST_A) & MASK;
    ulong z = (seed[6] ^ CONST_B) & MASK;
    ulong w = (seed[7] ^ CONST_C) & MASK;

    for (uint i = 0; i < 32; i++) {
        ulong key = seed[i & 7u];
        x = (x + rotl64(y ^ key, (i % 23u) + 5u)) & MASK;
        y = (y ^ rotl64((z + key + CONST_A) & MASK, (i % 19u) + 7u)) & MASK;
        z = (z + (w ^ key ^ CONST_B) + ((x * CONST_C) & MASK)) & MASK;
        w = rotl64((w + (x ^ y) + CONST_D) & MASK, (i % 17u) + 11u);

        x ^= x >> 29u;
        y ^= y >> 31u;
        z ^= z >> 33u;
        w ^= w >> 27u;
    }

    *h0 = (x ^ seed[0] ^ rotl64(z, 17u)) & MASK;
    *h1 = (y ^ seed[1] ^ rotl64(w, 29u)) & MASK;
    *h2 = (z ^ seed[2] ^ rotl64(x, 41u)) & MASK;
    *h3 = (w ^ seed[3] ^ rotl64(y, 53u)) & MASK;
}

__kernel void kk91_mine(
    __global const ulong *seed,
    __global const ulong *target,
    ulong nonce_base,
    __global uint *found_index
) {
    uint gid = get_global_id(0);

    if (*found_index != 0xFFFFFFFFu) {
        return;
    }

    ulong nonce = nonce_base + (ulong)gid;

    ulong h0;
    ulong h1;
    ulong h2;
    ulong h3;
    kkhash(nonce, seed, &h0, &h1, &h2, &h3);

    if (lt256(h0, h1, h2, h3, target)) {
        atomic_min(found_index, gid);
    }
}
"""


def opencl_gpu_available() -> bool:
    if cl is None or np is None:
        return False

    try:
        for platform in cl.get_platforms():
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                return True
    except Exception:
        return False
    return False


class OpenCLMiner:
    def __init__(self) -> None:
        if cl is None or np is None:
            raise RuntimeError("pyopencl and numpy are required for GPU mining")

        self.device = None
        for platform in cl.get_platforms():
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                self.device = devices[0]
                break

        if self.device is None:
            raise RuntimeError("No OpenCL GPU device found")

        self.context = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.context)
        self.program = cl.Program(self.context, KERNEL_SOURCE).build(options=["-cl-std=CL1.2"])
        self.kernel = cl.Kernel(self.program, "kk91_mine")

    def search_nonce(
        self,
        seed_words: tuple[int, int, int, int, int, int, int, int],
        target_words: tuple[int, int, int, int],
        start_nonce: int,
        batch_size: int = 131072,
        stop_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[int | None, int]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if start_nonce < 0 or start_nonce >= 1 << 64:
            raise ValueError("Nonce must be in uint64 range")

        mf = cl.mem_flags
        seed_host = np.array(seed_words, dtype=np.uint64)
        target_host = np.array(target_words, dtype=np.uint64)
        found_host = np.array([0xFFFFFFFF], dtype=np.uint32)

        seed_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=seed_host)
        target_buffer = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=target_host)
        found_buffer = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=found_host)

        attempts = 0
        nonce_base = int(start_nonce)
        started = time.perf_counter()

        while True:
            if stop_requested and stop_requested():
                return None, attempts

            found_host[0] = 0xFFFFFFFF
            cl.enqueue_copy(self.queue, found_buffer, found_host, is_blocking=True)

            self.kernel(
                self.queue,
                (batch_size,),
                None,
                seed_buffer,
                target_buffer,
                np.uint64(nonce_base),
                found_buffer,
            )
            cl.enqueue_copy(self.queue, found_host, found_buffer, is_blocking=True)

            attempts += batch_size
            elapsed = max(0.0001, time.perf_counter() - started)

            if progress_callback:
                progress_callback(
                    {
                        "backend": "gpu",
                        "nonce": nonce_base,
                        "attempts": attempts,
                        "elapsed": elapsed,
                        "hash_rate": attempts / elapsed,
                        "hash_preview": "gpu-batch",
                    }
                )

            if found_host[0] != 0xFFFFFFFF:
                return nonce_base + int(found_host[0]), attempts

            nonce_base += batch_size
            if nonce_base >= 1 << 64:
                nonce_base = 0
