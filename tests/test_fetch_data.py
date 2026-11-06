"""Small synthetic archives test input safety without network access."""
import hashlib
import importlib.util
from pathlib import Path
import stat
import zipfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "biomeme_fetch_data", Path(__file__).resolve().parents[1] / "scripts/fetch_data.py")
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)


def make_archive(tmp_path, *, symbolic=False, corrupt_entry=False):
    prefix = "BioMeme_4_22/software/BioMeme"
    payload = {"data/u2os_demo/example.bin": b"measurements\n",
               "statistics/core/expected.json": b"{}\n"}
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for index, (name, data) in enumerate(payload.items()):
            info = zipfile.ZipInfo(prefix + "/" + name)
            info.create_system = 3
            info.external_attr = ((stat.S_IFLNK if symbolic and index == 0 else stat.S_IFREG) | 0o644) << 16
            bundle.writestr(info, b"not the measurements" if corrupt_entry and index == 1 else data)
    files = [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
             for name, data in payload.items()]
    manifest = {"schema_version": 1, "source_doi": "10.5281/zenodo.24008759",
                "archive": {"filename": archive.name, "url": "https://example.invalid/release.zip",
                            "source_prefix": prefix, "bytes": archive.stat().st_size,
                            "sha256": F.digest(archive)}, "files": files}
    root = tmp_path / "repository"
    root.mkdir()
    return root, archive, manifest, payload


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "data/../outside",
                                 "data\\u2os_demo\\outside", "data//u2os_demo/x", "", "C:/outside"])
def test_unsafe_paths_rejected(name):
    with pytest.raises(F.DataError, match="Unsafe relative path"):
        F.safe_relative(name)


def test_archive_checksum_failure_installs_nothing(tmp_path):
    root, archive, manifest, _ = make_archive(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(F.DataError, match="Archive size or SHA-256"):
        F.install_payload(root, archive, manifest)
    assert list(root.iterdir()) == []


def test_existing_modified_input_prevents_any_installation(tmp_path):
    root, archive, manifest, _ = make_archive(tmp_path)
    existing = root / "statistics/core/expected.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep my edits")
    with pytest.raises(F.DataError, match="nothing was overwritten"):
        F.install_payload(root, archive, manifest)
    assert existing.read_text() == "keep my edits"
    assert not (root / "data").exists()


def test_install_is_idempotent_and_explicit_restore_works(tmp_path):
    root, archive, manifest, payload = make_archive(tmp_path)
    assert F.install_payload(root, archive, manifest) == 2
    timestamps = {name: (root / name).stat().st_mtime_ns for name in payload}
    assert F.install_payload(root, archive, manifest) == 0
    assert timestamps == {name: (root / name).stat().st_mtime_ns for name in payload}
    altered = root / next(iter(payload))
    altered.write_bytes(b"changed")
    assert F.install_payload(root, archive, manifest, replace=True) == 1
    assert F.input_status(root, manifest) == ([], [])
    for name, content in payload.items():
        assert (root / name).read_bytes() == content


def test_member_failure_leaves_all_inputs_uninstalled(tmp_path):
    root, archive, manifest, _ = make_archive(tmp_path, corrupt_entry=True)
    with pytest.raises(F.DataError, match="not the expected regular file|checksum"):
        F.install_payload(root, archive, manifest)
    assert list(root.iterdir()) == []


def test_symbolic_zip_entry_rejected(tmp_path):
    root, archive, manifest, _ = make_archive(tmp_path, symbolic=True)
    with pytest.raises(F.DataError, match="not the expected regular file"):
        F.install_payload(root, archive, manifest)
    assert list(root.iterdir()) == []


def test_symbolic_destination_rejected(tmp_path):
    root, archive, manifest, _ = make_archive(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "data").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(F.DataError, match="symbolic-link destination"):
        F.install_payload(root, archive, manifest)
    assert list(elsewhere.iterdir()) == []


def test_manifest_cannot_replace_scientific_code(tmp_path):
    _, _, manifest, _ = make_archive(tmp_path)
    manifest["files"][0]["path"] = "statistics/core/tahoe_statistics.py"
    with pytest.raises(F.DataError, match="executable source"):
        F.validate_manifest(manifest)
