"""File integrity verification and split-archive reconstruction."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

from src.config import BASE_DIR, RAW_CSV_FILENAME, RAW_SPLIT_FILES


def compute_sha256(file_path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of a local file in streaming chunks."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_split_checksums(base_dir: Path | str = BASE_DIR) -> dict[str, bool]:
    """Verify that all multipart zip archives match expected cryptographic digests."""
    base = Path(base_dir)
    results = {}
    for filename, expected_hash in RAW_SPLIT_FILES.items():
        file_path = base / filename
        if not file_path.exists():
            results[filename] = False
            continue
        actual_hash = compute_sha256(file_path)
        results[filename] = actual_hash == expected_hash
    return results


def reconstruct_zip_archive(
    base_dir: Path | str = BASE_DIR, output_zip: Path | str | None = None
) -> Path:
    """Concatenate multipart split volumes (data.z01, data.z02, data.zip) into a valid archive."""
    base = Path(base_dir)
    if output_zip is None:
        target = base / "combined.zip"
    else:
        target = Path(output_zip)

    # In PKZIP multi-volume format, joining parts in order (.z01, .z02, ... .zip)
    # allows modern unzip / python zipfile engines to unpack the central directory.
    parts = ["data.z01", "data.z02", "data.zip"]
    for part in parts:
        if not (base / part).exists():
            raise FileNotFoundError(f"Missing required multipart archive file: {base / part}")

    with open(target, "wb") as out_f:
        for part in parts:
            part_path = base / part
            with open(part_path, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f)

    # Verify that target is a readable zip file
    if not zipfile.is_zipfile(target):
        raise ValueError(f"Reconstructed file {target} is not recognized as a valid ZIP archive.")

    return target


def ensure_raw_csv_extracted(base_dir: Path | str = BASE_DIR) -> Path:
    """Ensure data.csv is present; if not, reconstruct archive and extract."""
    base = Path(base_dir)
    csv_path = base / RAW_CSV_FILENAME
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return csv_path

    combined_zip = base / "combined.zip"
    if not combined_zip.exists():
        combined_zip = reconstruct_zip_archive(base)

    with zipfile.ZipFile(combined_zip, "r") as zf:
        zf.extract(RAW_CSV_FILENAME, path=base)

    if not csv_path.exists():
        raise FileNotFoundError(f"Failed to extract {RAW_CSV_FILENAME} from {combined_zip}")

    return csv_path
