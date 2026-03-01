"""Tests for SPEC-W05: Wire reproducibility manifests into all entry points.

Verifies:
  - Each entry point imports build_run_manifest, write_run_manifest, verify_manifest
  - Each entry point defines a --verify-manifest CLI argument
  - build_run_manifest is called with correct run_type and config_snapshot
  - write_run_manifest writes a valid JSON manifest to the correct output directory
  - verify_manifest detects mismatches and reports them correctly
  - The manifest contains required fields: run_type, timestamp_utc, git_commit, config
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Root of the quant_engine project
PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Source-level checks: verify imports and argparse flags in all entry points
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntryPointImports:
    """Verify each entry point imports reproducibility functions."""

    ENTRY_POINTS = [
        "run_backtest.py",
        "run_train.py",
        "run_retrain.py",
        "run_autopilot.py",
        "run_predict.py",
    ]

    @pytest.mark.parametrize("filename", ENTRY_POINTS)
    def test_imports_build_run_manifest(self, filename):
        source = (PROJECT_ROOT / filename).read_text()
        assert "build_run_manifest" in source, (
            f"{filename} does not import build_run_manifest"
        )

    @pytest.mark.parametrize("filename", ENTRY_POINTS)
    def test_imports_write_run_manifest(self, filename):
        source = (PROJECT_ROOT / filename).read_text()
        assert "write_run_manifest" in source, (
            f"{filename} does not import write_run_manifest"
        )

    @pytest.mark.parametrize("filename", ENTRY_POINTS)
    def test_imports_verify_manifest(self, filename):
        source = (PROJECT_ROOT / filename).read_text()
        assert "verify_manifest" in source, (
            f"{filename} does not import verify_manifest"
        )


@pytest.mark.unit
class TestEntryPointArgparse:
    """Verify each entry point defines --verify-manifest in argparse."""

    ENTRY_POINTS = [
        "run_backtest.py",
        "run_train.py",
        "run_retrain.py",
        "run_autopilot.py",
        "run_predict.py",
    ]

    @pytest.mark.parametrize("filename", ENTRY_POINTS)
    def test_verify_manifest_flag_defined(self, filename):
        source = (PROJECT_ROOT / filename).read_text()
        assert "--verify-manifest" in source, (
            f"{filename} does not define --verify-manifest argument"
        )

    @pytest.mark.parametrize("filename", ENTRY_POINTS)
    def test_verify_manifest_flag_is_optional_string(self, filename):
        """The --verify-manifest flag should accept a path string and default to None."""
        source = (PROJECT_ROOT / filename).read_text()
        # Check that the flag uses type=str and default=None
        assert "default=None" in source, (
            f"{filename} --verify-manifest should default to None"
        )


# ---------------------------------------------------------------------------
# AST-level checks: verify build_run_manifest and write_run_manifest calls
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEntryPointCalls:
    """Verify entry points call build_run_manifest and write_run_manifest."""

    ENTRY_POINT_RUN_TYPES = {
        "run_backtest.py": "backtest",
        "run_train.py": "train",
        "run_retrain.py": "retrain",
        "run_autopilot.py": "autopilot",
        "run_predict.py": "predict",
    }

    @pytest.mark.parametrize("filename,expected_run_type", ENTRY_POINT_RUN_TYPES.items())
    def test_build_run_manifest_called_with_correct_run_type(self, filename, expected_run_type):
        """Verify build_run_manifest is called with the correct run_type string."""
        source = (PROJECT_ROOT / filename).read_text()
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match: build_run_manifest(...)
                if isinstance(func, ast.Name) and func.id == "build_run_manifest":
                    for kw in node.keywords:
                        if kw.arg == "run_type" and isinstance(kw.value, ast.Constant):
                            if kw.value.value == expected_run_type:
                                found = True
                                break
        assert found, (
            f"{filename} should call build_run_manifest(run_type='{expected_run_type}', ...)"
        )

    @pytest.mark.parametrize("filename", ENTRY_POINT_RUN_TYPES.keys())
    def test_write_run_manifest_called(self, filename):
        """Verify write_run_manifest is called in the entry point."""
        source = (PROJECT_ROOT / filename).read_text()
        tree = ast.parse(source)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "write_run_manifest":
                    found = True
                    break
        assert found, f"{filename} should call write_run_manifest(...)"


# ---------------------------------------------------------------------------
# Functional tests: build_run_manifest and write_run_manifest behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildRunManifest:
    """Test that build_run_manifest produces correct manifest dicts."""

    def test_manifest_has_required_fields(self):
        from quant_engine.reproducibility import build_run_manifest

        manifest = build_run_manifest(
            run_type="test",
            config_snapshot={"horizon": 10, "full": False},
        )

        assert manifest["run_type"] == "test"
        assert "timestamp_utc" in manifest
        assert "git_commit" in manifest
        assert "config" in manifest
        assert manifest["config"]["horizon"] == 10
        assert manifest["config"]["full"] is False

    def test_manifest_with_datasets(self):
        from quant_engine.reproducibility import build_run_manifest

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        manifest = build_run_manifest(
            run_type="train",
            config_snapshot={"mode": "full"},
            datasets={"training_data": df},
        )

        assert "datasets" in manifest
        assert manifest["datasets"]["training_data"]["rows"] == 3
        assert manifest["datasets"]["training_data"]["columns"] == 2
        assert "checksum" in manifest["datasets"]["training_data"]

    def test_manifest_with_extra_metadata(self):
        from quant_engine.reproducibility import build_run_manifest

        manifest = build_run_manifest(
            run_type="backtest",
            config_snapshot={},
            extra={"total_trades": 42, "sharpe_ratio": 1.5},
        )

        assert manifest["extra"]["total_trades"] == 42
        assert manifest["extra"]["sharpe_ratio"] == 1.5

    def test_manifest_handles_path_values_in_config(self):
        from quant_engine.reproducibility import build_run_manifest

        manifest = build_run_manifest(
            run_type="predict",
            config_snapshot={"output_dir": Path("/tmp/test")},
        )

        assert manifest["config"]["output_dir"] == "/tmp/test"


@pytest.mark.unit
class TestWriteRunManifest:
    """Test that write_run_manifest writes valid JSON to the correct path."""

    def test_writes_json_file(self, tmp_path):
        from quant_engine.reproducibility import build_run_manifest, write_run_manifest

        manifest = build_run_manifest(
            run_type="backtest",
            config_snapshot={"horizon": 10},
        )
        result_path = write_run_manifest(manifest, output_dir=tmp_path)

        assert result_path.exists()
        assert result_path.name.startswith("run_manifest_")
        assert result_path.name.endswith(".json")

        loaded = json.loads(result_path.read_text())
        assert loaded["run_type"] == "backtest"
        assert loaded["config"]["horizon"] == 10

        # Verify the "latest" symlink was created
        latest = tmp_path / "run_manifest_latest.json"
        assert latest.exists() or latest.is_symlink()

    def test_creates_output_dir_if_missing(self, tmp_path):
        from quant_engine.reproducibility import build_run_manifest, write_run_manifest

        nested = tmp_path / "deep" / "nested" / "dir"
        manifest = build_run_manifest(run_type="train", config_snapshot={})
        result_path = write_run_manifest(manifest, output_dir=nested)

        assert nested.exists()
        assert result_path.exists()

    def test_custom_filename(self, tmp_path):
        from quant_engine.reproducibility import build_run_manifest, write_run_manifest

        manifest = build_run_manifest(run_type="test", config_snapshot={})
        result_path = write_run_manifest(
            manifest, output_dir=tmp_path, filename="custom_manifest.json"
        )

        assert result_path.name == "custom_manifest.json"
        assert result_path.exists()


# ---------------------------------------------------------------------------
# Functional tests: verify_manifest behavior
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVerifyManifest:
    """Test that verify_manifest correctly detects matches and mismatches."""

    def test_verify_matching_config(self, tmp_path):
        from quant_engine.reproducibility import (
            build_run_manifest,
            verify_manifest,
            write_run_manifest,
        )

        config = {"horizon": 10, "full": False}
        manifest = build_run_manifest(run_type="test", config_snapshot=config)
        manifest_path = write_run_manifest(manifest, output_dir=tmp_path)

        result = verify_manifest(manifest_path, config_snapshot=config)

        assert result["config_match"] is True
        assert result["git_match"] is True

    def test_verify_detects_config_mismatch(self, tmp_path):
        from quant_engine.reproducibility import (
            build_run_manifest,
            verify_manifest,
            write_run_manifest,
        )

        original_config = {"horizon": 10, "full": False}
        manifest = build_run_manifest(run_type="test", config_snapshot=original_config)
        manifest_path = write_run_manifest(manifest, output_dir=tmp_path)

        changed_config = {"horizon": 20, "full": True}
        result = verify_manifest(manifest_path, config_snapshot=changed_config)

        assert result["config_match"] is False
        assert len(result["mismatches"]) > 0

    def test_verify_missing_manifest_file(self, tmp_path):
        from quant_engine.reproducibility import verify_manifest

        result = verify_manifest(tmp_path / "nonexistent.json")

        assert result["git_match"] is False
        assert result["config_match"] is False
        assert result["data_match"] is False
        assert any("not found" in m for m in result["mismatches"])

    def test_verify_data_checksum_mismatch(self, tmp_path):
        from quant_engine.reproducibility import (
            build_run_manifest,
            verify_manifest,
            write_run_manifest,
        )

        df_original = pd.DataFrame({"a": [1, 2, 3]})
        manifest = build_run_manifest(
            run_type="test",
            config_snapshot={},
            datasets={"data": df_original},
        )
        manifest_path = write_run_manifest(manifest, output_dir=tmp_path)

        df_changed = pd.DataFrame({"a": [10, 20, 30]})
        result = verify_manifest(
            manifest_path,
            config_snapshot={"_datasets": {"data": df_changed}},
        )

        assert result["data_match"] is False


# ---------------------------------------------------------------------------
# CLI flag smoke tests: --verify-manifest parses correctly
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCLIFlagParsing:
    """Verify --verify-manifest parses correctly for each entry point."""

    ENTRY_POINTS = [
        "run_backtest.py",
        "run_train.py",
        "run_retrain.py",
        "run_autopilot.py",
        "run_predict.py",
    ]

    @pytest.mark.parametrize("filename", ENTRY_POINTS)
    def test_help_includes_verify_manifest(self, filename):
        """Run --help and verify --verify-manifest appears in the output."""
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / filename), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"{filename} --help failed: {result.stderr}"
        assert "--verify-manifest" in result.stdout, (
            f"{filename} --help does not mention --verify-manifest"
        )


# ---------------------------------------------------------------------------
# Integration test: manifest round-trip (build → write → verify)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestManifestRoundTrip:
    """Full round-trip: build, write, and verify a manifest."""

    def test_round_trip_matches(self, tmp_path):
        from quant_engine.reproducibility import (
            build_run_manifest,
            verify_manifest,
            write_run_manifest,
        )

        config = {
            "horizon": 10,
            "full": False,
            "tickers": None,
            "output": None,
            "quiet": True,
        }
        df = pd.DataFrame({"close": [100, 101, 102], "volume": [1e6, 1.1e6, 0.9e6]})

        manifest = build_run_manifest(
            run_type="backtest",
            config_snapshot=config,
            datasets={"price_data": df},
            extra={"total_trades": 15, "sharpe_ratio": 1.2},
        )
        manifest_path = write_run_manifest(manifest, output_dir=tmp_path)

        # Verify against the same config + datasets
        result = verify_manifest(
            manifest_path,
            config_snapshot={**config, "_datasets": {"price_data": df}},
        )

        assert result["git_match"] is True
        assert result["config_match"] is True
        assert result["data_match"] is True
        assert result["mismatches"] == []

    def test_manifest_json_is_human_readable(self, tmp_path):
        """Verify the manifest is pretty-printed JSON (indented)."""
        from quant_engine.reproducibility import build_run_manifest, write_run_manifest

        manifest = build_run_manifest(
            run_type="predict",
            config_snapshot={"horizon": 5},
        )
        manifest_path = write_run_manifest(manifest, output_dir=tmp_path)

        raw = manifest_path.read_text()
        # Pretty-printed JSON has newlines and indentation
        assert "\n" in raw
        assert "  " in raw

        # Verify it parses back to the same data
        loaded = json.loads(raw)
        assert loaded["run_type"] == "predict"
        assert loaded["config"]["horizon"] == 5
