#!/usr/bin/env python3
"""Fetch and verify the immutable inputs for the BioMeme reproduction example.

Only the exact files listed in data_manifest.json are extracted. Scientific
code is never downloaded or replaced by this helper. No third-party Python
packages are needed; curl is preferred when available, with TLS verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("data_manifest.json")
CHUNK = 1024 * 1024
ALLOWED_PREFIXES = ("data/u2os_demo/", "statistics/core/", "statistics/a5/")


class DataError(RuntimeError):
    """A download, provenance, or local input integrity check failed."""


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        if not new_url.startswith("https://"):
            raise DataError("The download redirected to an insecure URL")
        return super().redirect_request(request, response, code, message, headers, new_url)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            value.update(block)
    return value.hexdigest()


def safe_relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise DataError(f"Unsafe relative path: {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise DataError(f"Unsafe relative path: {name!r}")
    result = PurePosixPath(name)
    if result.is_absolute():
        raise DataError(f"Unsafe relative path: {name!r}")
    return result


def destination(root: Path, name: str) -> Path:
    """Resolve a permitted input path without following local symbolic links."""
    relative = safe_relative(name)
    if not any(name.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        raise DataError(f"Path is outside the data allowlist: {name}")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise DataError(f"Refusing to use a symbolic-link destination: {name}")
    return path


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise DataError("Unsupported input manifest version")
    archive = manifest["archive"]
    safe_relative(archive["filename"])
    safe_relative(archive["source_prefix"])
    if "/" in archive["filename"] or not archive["url"].startswith("https://"):
        raise DataError("Invalid archive source")
    names = set()
    for entry in manifest["files"]:
        name = entry["path"]
        safe_relative(name)
        if not any(name.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            raise DataError(f"Path is outside the data allowlist: {name}")
        if name.endswith(".py") or name in names:
            raise DataError(f"Duplicate path or executable source in data manifest: {name}")
        names.add(name)
        for item in (entry, archive):
            if not isinstance(item["bytes"], int) or item["bytes"] < 0:
                raise DataError("Invalid expected byte count")
            if len(item["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in item["sha256"]):
                raise DataError("Invalid expected SHA-256")
    if not names:
        raise DataError("Empty input manifest")


def matches(path: Path, entry: dict) -> bool:
    return (path.is_file() and not path.is_symlink()
            and path.stat().st_size == entry["bytes"]
            and digest(path) == entry["sha256"])


def input_status(root: Path, manifest: dict) -> tuple[list[str], list[str]]:
    missing, changed = [], []
    for entry in manifest["files"]:
        path = destination(root, entry["path"])
        if not path.exists():
            missing.append(entry["path"])
        elif not matches(path, entry):
            changed.append(entry["path"])
    return missing, changed


def verify_archive(archive: Path, metadata: dict) -> None:
    if not matches(archive, metadata):
        raise DataError(f"Archive size or SHA-256 does not match the published release: {archive}")


def download_archive(metadata: dict, cache: Path, *, replace: bool = False) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / metadata["filename"]
    if target.exists() or target.is_symlink():
        if matches(target, metadata):
            return target
        if not replace:
            raise DataError(f"Cached archive differs from the release: {target}. "
                            "Remove it, or use --replace to download it again.")
    print(f"Downloading {metadata['bytes'] / 1_000_000:.1f} MB from Zenodo; "
          "the archive is checked before any input is installed.", flush=True)
    descriptor, temporary = tempfile.mkstemp(prefix="download-", suffix=".part", dir=cache)
    os.close(descriptor)
    part = Path(temporary)
    try:
        curl = shutil.which("curl")
        if curl:
            subprocess.run([
                curl, "--fail", "--location", "--silent", "--show-error",
                "--proto", "=https", "--proto-redir", "=https", "--retry", "3",
                "--connect-timeout", "30", "--max-time", "1800",
                "--output", str(part), metadata["url"],
            ], check=True)
        else:
            # urllib's default SSL context verifies both certificates and names.
            request = urllib.request.Request(metadata["url"], headers={"User-Agent": "BioMeme-data/4.22"})
            opener = urllib.request.build_opener(HTTPSOnlyRedirect())
            with opener.open(request, timeout=120) as response, part.open("wb") as handle:
                shutil.copyfileobj(response, handle, CHUNK)
        verify_archive(part, metadata)
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)
    return target


def install_payload(root: Path, archive: Path, manifest: dict, *, replace: bool = False) -> int:
    """Validate all requested entries before installing any; commit files atomically.

    A stopped installation may contain a subset of verified files. A later call
    skips those files and resumes safely. Existing modified inputs are preserved
    unless the caller explicitly requests replacement.
    """
    validate_manifest(manifest)
    missing, changed = input_status(root, manifest)
    if changed and not replace:
        raise DataError("Local input files differ from the release; nothing was overwritten: "
                        + ", ".join(changed) + ". Use --replace only to restore release inputs.")
    needed = set(missing + changed)
    if not needed:
        return 0
    verify_archive(archive, manifest["archive"])
    with tempfile.TemporaryDirectory(prefix=".biomeme-inputs-", dir=root) as temporary:
        stage = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            infos = {}
            for info in bundle.infolist():
                if info.filename in infos:
                    raise DataError(f"Duplicate ZIP member: {info.filename}")
                infos[info.filename] = info
            for entry in manifest["files"]:
                if entry["path"] not in needed:
                    continue
                member = manifest["archive"]["source_prefix"] + "/" + entry["path"]
                info = infos.get(member)
                if info is None:
                    raise DataError(f"Expected input is absent from the archive: {entry['path']}")
                kind = stat.S_IFMT(info.external_attr >> 16)
                if info.is_dir() or kind not in (0, stat.S_IFREG) or info.file_size != entry["bytes"]:
                    raise DataError(f"ZIP entry is not the expected regular file: {entry['path']}")
                staged = stage.joinpath(*safe_relative(entry["path"]).parts)
                staged.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, staged.open("xb") as handle:
                    shutil.copyfileobj(source, handle, CHUNK)
                if not matches(staged, entry):
                    raise DataError(f"Input checksum does not match: {entry['path']}")
        installed = 0
        for entry in manifest["files"]:
            if entry["path"] not in needed:
                continue
            target = destination(root, entry["path"])
            if target.exists():
                if matches(target, entry):
                    continue
                if not replace:
                    raise DataError(f"Local input changed during extraction; preserved: {entry['path']}")
                if not target.is_file():
                    raise DataError(f"Destination is not a regular file: {entry['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            destination(root, entry["path"])
            os.replace(stage.joinpath(*safe_relative(entry["path"]).parts), target)
            installed += 1
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded release ZIP; no network access")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "biomeme",
                        help="Download cache (default: repository .cache/biomeme)")
    parser.add_argument("--check", action="store_true", help="Verify installed inputs without downloading or changing files")
    parser.add_argument("--replace", action="store_true", help="Restore modified local inputs and an invalid download cache")
    args = parser.parse_args(argv)
    if args.check and (args.archive or args.replace):
        parser.error("--check cannot be combined with --archive or --replace")
    try:
        manifest = json.loads(MANIFEST.read_text())
        validate_manifest(manifest)
        missing, changed = input_status(ROOT, manifest)
        if args.check:
            if missing or changed:
                raise DataError(f"Input verification failed: {len(missing)} missing, {len(changed)} changed. "
                                + ", ".join(missing + changed))
        elif missing or changed:
            if changed and not args.replace:
                raise DataError("Local inputs have changed; nothing was overwritten: " + ", ".join(changed)
                                + ". Use --replace only to restore release inputs.")
            archive = args.archive.resolve() if args.archive else download_archive(
                manifest["archive"], args.cache_dir.resolve(), replace=args.replace)
            installed = install_payload(ROOT, archive, manifest, replace=args.replace)
            print(f"Installed {installed} verified input files.")
        print(f"Verified {len(manifest['files'])} input files "
              f"({sum(e['bytes'] for e in manifest['files']) / 1_000_000:.1f} MB) "
              f"from {manifest['source_doi']}.")
        return 0
    except (DataError, OSError, ValueError, KeyError, zipfile.BadZipFile,
            subprocess.CalledProcessError) as error:
        print(f"BioMeme input setup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
