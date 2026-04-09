"""Tests for hardware-aware auto configuration."""

from __future__ import annotations

import importlib

import framex as fx

cfgmod = importlib.import_module("framex.config")


class TestHardwareAutoConfig:
    def test_recommend_prefers_hpc_when_cluster_env_present(self, monkeypatch):
        monkeypatch.setenv("FRAMEX_DASK_SCHEDULER_ADDRESS", "tcp://scheduler:8786")
        monkeypatch.setattr(cfgmod, "_detect_total_memory_gb", lambda: 64.0)
        monkeypatch.setattr(cfgmod, "_detect_gpu_array_backend", lambda: None)
        monkeypatch.setattr(cfgmod, "_module_available", lambda name: name == "numexpr")

        cfg = fx.recommend_best_performance_config()
        assert cfg.backend == "hpc"
        assert cfg.partition_size_rows == 1_000_000
        assert cfg.array_backend == "numexpr"

    def test_recommend_prefers_gpu_backend_when_available(self, monkeypatch):
        monkeypatch.delenv("FRAMEX_DASK_SCHEDULER_ADDRESS", raising=False)
        monkeypatch.delenv("FRAMEX_RAY_ADDRESS", raising=False)
        monkeypatch.delenv("FRAMEX_DASK_SLURM", raising=False)
        monkeypatch.setattr(cfgmod, "_detect_total_memory_gb", lambda: 128.0)
        monkeypatch.setattr(cfgmod, "_detect_gpu_array_backend", lambda: "cupy")
        monkeypatch.setattr(cfgmod, "_module_available", lambda name: False)

        cfg = fx.recommend_best_performance_config()
        assert cfg.backend == "threads"
        assert cfg.partition_size_rows == 2_000_000
        assert cfg.array_backend == "cupy"

    def test_auto_configure_hardware_apply_false_does_not_mutate_global(self, monkeypatch):
        original = fx.get_config()
        monkeypatch.setattr(cfgmod, "_detect_total_memory_gb", lambda: 8.0)
        monkeypatch.setattr(cfgmod, "_detect_gpu_array_backend", lambda: None)
        monkeypatch.setattr(cfgmod, "_module_available", lambda name: False)

        suggested = fx.auto_configure_hardware(apply=False)
        current = fx.get_config()

        assert suggested.partition_size_rows == 250_000
        assert current == original

    def test_auto_configure_hardware_apply_true_updates_global(self, monkeypatch):
        prev = fx.get_config()
        monkeypatch.setattr(cfgmod, "_detect_total_memory_gb", lambda: 16.0)
        monkeypatch.setattr(cfgmod, "_detect_gpu_array_backend", lambda: None)
        monkeypatch.setattr(cfgmod, "_module_available", lambda name: name == "numba")

        applied = fx.auto_configure_hardware(apply=True)
        current = fx.get_config()

        assert applied == current
        assert current.partition_size_rows == 500_000
        assert current.array_backend == "numba"

        # Restore previous config to avoid cross-test coupling.
        monkeypatch.setattr(cfgmod, "_current", prev)
