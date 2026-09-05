"""vllm-metamodels: same-host zero-copy transport through POSIX shared memory (1.1.0.post7).

vLLM's ``collective_rpc`` pickles every argument / return value and ships the bytes over
ZMQ between the client process and the engine-core / worker processes.  For the fork's
bulk payloads -- captured activations (``[requests x positions x hidden]`` bf16, ~1.2 GB for
1,024 texts of the 27B) and the per-call steering / readout vector blocks -- that means
three full copies (pickle, socket, unpickle) plus Python overhead.  When both sides run on
the same host (offline ``LLM``, single-node ``vllm serve``) the payload can instead be
written ONCE into a named shared-memory segment and the RPC carries only a descriptor
(name, offsets, shapes, dtypes: a few hundred bytes).

``put(tensors)`` creates the segment, copies each tensor in and returns the descriptor;
``get(desc)`` attaches, unlinks the name (the mapping survives until every process drops
it) and returns tensor VIEWS into the mapping plus the ``SharedMemory`` handle that keeps
it alive.  ``copy=True`` returns ordinary tensors instead (one memcpy, no lifetime coupling).
Enabled per direction by the plugin (``VLLM_LENS_SHM=1`` -> copy-out, ``=view`` -> views);
any failure to attach falls back to the pickled path.
"""

from __future__ import annotations

import logging
import mmap
import os
import socket
import uuid
from typing import Any

import numpy as np
import torch

try:  # CPython's POSIX shm binding (what multiprocessing.shared_memory uses underneath); we drive it
    import _posixshmem  # type: ignore
except ImportError:  # pragma: no cover - non-POSIX
    _posixshmem = None

logger = logging.getLogger(__name__)

_DTYPES = {str(dt).replace("torch.", ""): dt for dt in (
    torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool
)}


def host_id() -> str:
    """Identity of this host + shared-memory namespace (a descriptor from another host is refused)."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot = f.read().strip()
    except Exception:  # noqa: BLE001
        boot = ""
    return f"{socket.gethostname()}:{boot}"


class _Segment:
    """A POSIX shared-memory segment mapped into this process.  Deliberately NOT
    ``multiprocessing.shared_memory.SharedMemory``: Python <= 3.12 registers every attach
    with the resource tracker and unlinks it at exit (bpo-39959), and unregistering a name
    the tracker never saw prints tracebacks from the tracker process.  The lifetime is ours:
    the producer creates + unlinks-on-consume, consumers keep the mapping alive as ``handle``."""

    def __init__(self, name: str, size: int, create: bool):
        if _posixshmem is None:
            raise RuntimeError("POSIX shared memory is unavailable on this platform")
        self.name = name
        flags = (os.O_CREAT | os.O_EXCL | os.O_RDWR) if create else os.O_RDWR
        fd = _posixshmem.shm_open(name, flags, mode=0o600)
        try:
            if create:
                os.ftruncate(fd, size)
            self.size = os.fstat(fd).st_size
            self.buf = mmap.mmap(fd, self.size)
        finally:
            os.close(fd)

    def unlink(self) -> None:
        try:
            _posixshmem.shm_unlink(self.name)
        except FileNotFoundError:
            pass

    def close(self) -> None:
        try:
            self.buf.close()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:  # pragma: no cover - lifetime helper
        self.close()


def put(tensors: dict[str, torch.Tensor], tag: str = "lens") -> dict[str, Any]:
    """Copy CPU tensors into ONE new shared-memory segment; return its picklable descriptor."""
    segs: list[tuple[str, str, tuple[int, ...], int, int]] = []
    off = 0
    for key, t in tensors.items():
        n = t.numel() * t.element_size()
        off = (off + 63) // 64 * 64
        segs.append((key, str(t.dtype).replace("torch.", ""), tuple(t.shape), off, n))
        off += n
    nbytes = max(off, 1)
    name = f"/vllm_lens_{tag}_{os.getpid()}_{uuid.uuid4().hex[:10]}"
    shm = _Segment(name, nbytes, create=True)
    try:
        buf = torch.frombuffer(shm.buf, dtype=torch.uint8, count=nbytes) if nbytes else None
        for key, _dt, _shape, o, n in segs:
            if n == 0:
                continue
            src = tensors[key].detach()
            if src.device.type != "cpu":
                src = src.cpu()
            src = src.contiguous()
            buf[o : o + n].copy_(src.view(-1).view(torch.uint8))  # type: ignore[union-attr]
        del buf
    except Exception:
        shm.close()
        shm.unlink()
        raise
    shm.close()  # the consumer unlinks the name
    return {"shm": name, "nbytes": nbytes, "segments": segs, "host": host_id()}


def get(desc: dict[str, Any], copy: bool = False, unlink: bool = True) -> tuple[dict[str, torch.Tensor], Any]:
    """Attach to a segment from :func:`put`; return ``(tensors, handle)``.  ``tensors`` are views
    into the mapping unless ``copy`` (then ``handle`` is None and the mapping is released).
    Keep ``handle`` alive as long as the views are used.  Raises if the segment cannot be
    opened on this host (caller falls back to the pickled payload)."""
    if desc.get("host") != host_id():
        raise RuntimeError(f"shared-memory descriptor from another host ({desc.get('host')} != {host_id()})")
    shm = _Segment(desc["shm"], int(desc["nbytes"]), create=False)
    try:
        if unlink:
            shm.unlink()  # the mapping stays valid until every process drops it
        buf = torch.frombuffer(shm.buf, dtype=torch.uint8, count=int(desc["nbytes"]))
        out: dict[str, torch.Tensor] = {}
        for key, dt, shape, o, n in desc["segments"]:
            dtype = _DTYPES[dt]
            if n == 0:
                out[key] = torch.empty(tuple(shape), dtype=dtype)
                continue
            t = buf[o : o + n].view(dtype).view(*shape)
            out[key] = t.clone() if copy else t
        if copy:
            del buf
            shm.close()
            return out, None
        return out, shm
    except Exception:
        shm.close()
        raise


def release(handle: Any) -> None:
    """Drop the mapping once every view is dead (also what ``SharedMemory.__del__`` does)."""
    if handle is not None:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass


def shm_mode() -> str:
    """``""`` (off) | ``"copy"`` | ``"view"`` from ``VLLM_LENS_SHM``."""
    v = os.environ.get("VLLM_LENS_SHM", "").strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return ""
    return "view" if v == "view" else "copy"


def total_bytes(desc: dict[str, Any]) -> int:
    return int(desc.get("nbytes", 0))


def as_uint8_numpy(t: torch.Tensor) -> np.ndarray:  # pragma: no cover - debugging aid
    return t.view(torch.uint8).numpy()
