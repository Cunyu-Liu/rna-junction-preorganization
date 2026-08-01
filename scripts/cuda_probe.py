#!/usr/bin/env python3
"""Fail-closed CUDA hardware probe; no project data or model code is loaded."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised on remote envs
        print(json.dumps({"status": "BLOCKED_TORCH_IMPORT", "error": repr(exc)}))
        return 2

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not torch.cuda.is_available():
        print(
            json.dumps(
                {
                    "status": "FAILED_CUDA_UNAVAILABLE",
                    "cuda_visible_devices": visible,
                    "torch_version": torch.__version__,
                },
                sort_keys=True,
            )
        )
        return 3

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    probe = torch.ones((1,), device=device)
    if probe.device.type != "cuda":
        print(
            json.dumps(
                {
                    "status": "FAILED_CPU_FALLBACK",
                    "actual_device": str(probe.device),
                    "cuda_visible_devices": visible,
                },
                sort_keys=True,
            )
        )
        return 4
    torch.cuda.synchronize(device)
    print(
        json.dumps(
            {
                "status": "CUDA_PROBE_PASS",
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_visible_devices": visible,
                "logical_device": str(device),
                "device_name": props.name,
                "total_memory_bytes": props.total_memory,
                "probe_tensor_device": str(probe.device),
                "probe_tensor_value": float(probe.item()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
