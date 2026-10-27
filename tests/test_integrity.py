"""Tests for file checksums and archive integrity."""

from pathlib import Path

from src.config import BASE_DIR
from src.integrity import compute_sha256, verify_split_checksums


def test_multipart_split_checksums() -> None:
    """Verify that all multipart zip parts exist and match expected SHA-256."""
    results = verify_split_checksums(BASE_DIR)
    for filename, is_valid in results.items():
        assert is_valid, f"SHA-256 checksum mismatch for {filename}"


def test_compute_sha256_known_string(tmp_path: Path) -> None:
    """Verify SHA-256 calculation logic against a known hash."""
    test_file = tmp_path / "hello.txt"
    test_file.write_text("electricity_theft_detection_2018")
    expected = "b1571c38dc3235968dcfac24e643400aa68cbfce48ccf8cd5f4d49f4c4d550e7"
    assert compute_sha256(test_file) == expected
