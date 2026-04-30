from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meteogram2zarr.cli import parse_args  # noqa: E402


def test_parse_help_options() -> None:
    args = parse_args(
        [
            "--input-dir",
            "input",
            "--output",
            "out.zarr",
            "--max-time",
            "3",
            "--experiments",
            "20260304110254",
        ]
    )
    assert args.input_dir == "input"
    assert args.output == "out.zarr"
    assert args.max_time == 3


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(["--help"])
    assert exc.value.code == 0
    assert "Convert COSMO-SPECS" in capsys.readouterr().out
