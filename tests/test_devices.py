from __future__ import annotations

import unittest

from whisper_large_v3.devices import choose_device, pipeline_device


class _Cuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _Mps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _Backends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _Mps(mps_available)


class _Torch:
    def __init__(self, cuda_available: bool = False, mps_available: bool = False) -> None:
        self.cuda = _Cuda(cuda_available)
        self.backends = _Backends(mps_available)

    def device(self, name: str) -> str:
        return f"device:{name}"


class DeviceTests(unittest.TestCase):
    def test_choose_device_prefers_cuda_then_mps_then_cpu(self) -> None:
        self.assertEqual(choose_device(_Torch(cuda_available=True), "auto"), "cuda")
        self.assertEqual(choose_device(_Torch(mps_available=True), "auto"), "mps")
        self.assertEqual(choose_device(_Torch(), "auto"), "cpu")

    def test_choose_device_rejects_unavailable_requested_device(self) -> None:
        with self.assertRaises(SystemExit):
            choose_device(_Torch(), "cuda")
        with self.assertRaises(SystemExit):
            choose_device(_Torch(), "mps")

    def test_pipeline_device_maps_transformers_device_values(self) -> None:
        torch = _Torch()

        self.assertEqual(pipeline_device(torch, "cuda"), 0)
        self.assertEqual(pipeline_device(torch, "mps"), "device:mps")
        self.assertEqual(pipeline_device(torch, "cpu"), -1)


if __name__ == "__main__":
    unittest.main()
