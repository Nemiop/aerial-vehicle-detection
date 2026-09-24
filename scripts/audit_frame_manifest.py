"""Validate and enrich the selected-frame manifest without touching images."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_DIRECTORY / "Data" / "Frames" / "frames_manifest.csv"
AUDIT_PATH = PROJECT_DIRECTORY / "My Artifacts" / "input_audit.json"
TEMP_MANIFEST_PATH = MANIFEST_PATH.with_suffix(".csv.tmp")
FRAME_NAME_PATTERN = re.compile(r"^(train_[A-D])_frame_(\d{8})\.jpg$")
EXPECTED_SPLIT_BY_VIDEO = {
    "train_A.mp4": "train",
    "train_B.mp4": "train",
    "train_C.mp4": "train",
    "train_D.mp4": "validation",
}
ENRICHED_FIELDS = (
    "frame_key",
    "relative_image_path",
    "image_path",
    "image_sha256",
    "source_video_relative_path",
    "source_video_path",
    "source_video_sha256",
    "source_frame_id",
    "timestamp_seconds",
    "width",
    "height",
    "split",
)


def calculate_sha256(file_path: Path) -> str:
    """Calculate a file digest using bounded memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(stored_path: str, relative_path: str | None = None) -> Path:
    """Resolve an existing absolute path or a path relative to the project."""
    if relative_path:
        candidate = PROJECT_DIRECTORY / Path(relative_path)
        if candidate.is_file():
            return candidate.resolve()
    candidate = Path(stored_path)
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(stored_path)


def read_manifest() -> list[dict[str, str]]:
    """Read all selected frames from the canonical manifest."""
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise ValueError("Frame manifest is empty")
    return rows


def enrich_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict]:
    """Validate provenance and add portable identifiers and hashes."""
    enriched_rows: list[dict[str, object]] = []
    video_hashes: dict[Path, str] = {}
    frame_keys: set[str] = set()
    image_paths: set[Path] = set()

    for row in rows:
        image_path = resolve_manifest_path(
            row["image_path"], row.get("relative_image_path")
        )
        source_video_path = resolve_manifest_path(
            row["source_video_path"], row.get("source_video_relative_path")
        )
        expected_split = EXPECTED_SPLIT_BY_VIDEO.get(source_video_path.name)
        if expected_split is None:
            raise ValueError(f"Unexpected source video: {source_video_path.name}")
        if row["split"] != expected_split:
            raise ValueError(
                f"Wrong split for {source_video_path.name}: {row['split']}"
            )

        name_match = FRAME_NAME_PATTERN.fullmatch(image_path.name)
        if not name_match:
            raise ValueError(f"Unexpected frame filename: {image_path.name}")
        source_frame_id = int(row["source_frame_id"])
        if int(name_match.group(2)) != source_frame_id:
            raise ValueError(f"Frame ID mismatch: {image_path}")
        if name_match.group(1) != source_video_path.stem:
            raise ValueError(f"Video name mismatch: {image_path}")

        frame_key = f"{source_video_path.stem}:{source_frame_id:08d}"
        if frame_key in frame_keys or image_path in image_paths:
            raise ValueError(f"Duplicate frame: {frame_key}")
        frame_keys.add(frame_key)
        image_paths.add(image_path)

        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        if source_video_path not in video_hashes:
            video_hashes[source_video_path] = calculate_sha256(source_video_path)

        enriched_rows.append(
            {
                "frame_key": frame_key,
                "relative_image_path": image_path.relative_to(PROJECT_DIRECTORY).as_posix(),
                "image_path": str(image_path),
                "image_sha256": calculate_sha256(image_path),
                "source_video_relative_path": source_video_path.relative_to(
                    PROJECT_DIRECTORY
                ).as_posix(),
                "source_video_path": str(source_video_path),
                "source_video_sha256": video_hashes[source_video_path],
                "source_frame_id": source_frame_id,
                "timestamp_seconds": row["timestamp_seconds"],
                "width": width,
                "height": height,
                "split": row["split"],
            }
        )

    split_counts: dict[str, int] = {}
    video_counts: dict[str, int] = {}
    for row in enriched_rows:
        split_counts[str(row["split"])] = split_counts.get(str(row["split"]), 0) + 1
        video_name = Path(str(row["source_video_path"])).name
        video_counts[video_name] = video_counts.get(video_name, 0) + 1
    audit = {
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256_before_enrichment": calculate_sha256(MANIFEST_PATH),
        "frame_count": len(enriched_rows),
        "split_counts": split_counts,
        "video_counts": video_counts,
        "unique_frame_count": len(frame_keys),
        "all_files_exist": True,
    }
    return enriched_rows, audit


def write_manifest(rows: list[dict[str, object]]) -> None:
    """Atomically replace the manifest with its validated enriched form."""
    with TEMP_MANIFEST_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ENRICHED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    TEMP_MANIFEST_PATH.replace(MANIFEST_PATH)


def write_audit(audit: dict) -> None:
    """Save the audit summary outside the dataset."""
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit["manifest_sha256_after_enrichment"] = calculate_sha256(MANIFEST_PATH)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Validate all selected frames and write portable manifest metadata."""
    rows, audit = enrich_rows(read_manifest())
    write_manifest(rows)
    write_audit(audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

