#!/usr/bin/env python3
"""Fail-closed source, package, and binary residue audit for Emelex releases."""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable

READ_SIZE = 1 << 20
MAX_MANIFEST_BYTES = 1 << 20
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_FILE_BYTES = 1 << 30
MAX_ARCHIVE_TOTAL_BYTES = 4 << 30
IGNORED_WORKSPACE_ROOTS = {".git", "target"}
FORBIDDEN_GENERATED_COMPONENTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "target",
}
FORBIDDEN_GENERATED_NAMES = {".DS_Store"}
APPLE_SILICON_TARGET = "aarch64-apple-darwin"
CARGO_GENERATED_MEMBERS = {"Cargo.toml", "Cargo.toml.orig", ".cargo_vcs_info.json"}
MACHO_HEADER_BYTES = 32
MAX_MACHO_LOAD_COMMANDS = 4096
MAX_MACHO_LOAD_COMMAND_BYTES = 16 << 20
LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32
LC_LOAD_DYLIB = 0x0C
LC_LOAD_WEAK_DYLIB = 0x80000018
LC_REEXPORT_DYLIB = 0x8000001F
LC_LAZY_LOAD_DYLIB = 0x20
LC_LOAD_UPWARD_DYLIB = 0x80000023
DYLIB_LOAD_COMMANDS = {
    LC_LOAD_DYLIB,
    LC_LOAD_WEAK_DYLIB,
    LC_REEXPORT_DYLIB,
    LC_LAZY_LOAD_DYLIB,
    LC_LOAD_UPWARD_DYLIB,
}
SYSTEM_DYLIB_PREFIXES = ("/usr/lib/", "/System/Library/")
PLATFORM_MACOS = 1
RELEASE_MINIMUM_MACOS = (26, 5, 0)


class AuditFailure(ValueError):
    """A release input contains residue or cannot be audited safely."""


@dataclass(frozen=True)
class Marker:
    label: str
    value: bytes


@dataclass(frozen=True)
class PackageIdentity:
    name: str
    version: str

    @property
    def archive_root(self) -> str:
        return f"{self.name}-{self.version}"


def residue_markers(repository: Path) -> tuple[Marker, ...]:
    """Construct forbidden markers without embedding them in this source file."""
    former_name = b"nit" + b"pin"
    former_dashed_name = b"nit" + b"-" + b"pin"
    private_home_prefix = b"/" + b"Users" + b"/"
    dynamic = [
        Marker("former product name", former_name),
        Marker("former dashed product name", former_dashed_name),
        Marker("private macOS home prefix", private_home_prefix),
        Marker("source checkout path", os.fsencode(repository.resolve())),
    ]
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None
    if home is not None and home != Path("/"):
        dynamic.append(Marker("builder home path", os.fsencode(home)))

    unique: dict[bytes, Marker] = {}
    for marker in dynamic:
        normalized = marker.value.lower()
        if marker.label in {"source checkout path", "builder home path"}:
            normalized = normalized.rstrip(b"/")
        if normalized:
            unique.setdefault(normalized, Marker(marker.label, normalized))
    return tuple(unique.values())


def audit_repository(repository: Path) -> int:
    """Audit every Git-tracked filename and regular-file content."""
    root = repository.resolve()
    _audit_workspace_generated_residue(root)
    result = subprocess.run(
        ["git", "-C", os.fspath(root), "ls-files", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AuditFailure("cannot enumerate tracked repository files")
    raw_names = [name for name in result.stdout.split(b"\0") if name]
    if not raw_names:
        raise AuditFailure("repository has no tracked files")

    markers = residue_markers(root)
    for raw_name in raw_names:
        try:
            relative = raw_name.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AuditFailure("tracked filename is not UTF-8") from error
        path = _safe_relative_path(relative, "tracked filename")
        _audit_generated_path(path, "tracked filename", allow_worktree_sentinel=True)
        _scan_name(relative, f"tracked filename {relative!r}", markers)
        absolute = root.joinpath(*path.parts)
        metadata = absolute.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise AuditFailure(f"tracked path is not a regular file: {relative}")
        with absolute.open("rb") as stream:
            _scan_stream(stream, f"tracked file {relative}", markers)
    return len(raw_names)


def audit_crate(crate: Path, repository: Path) -> int:
    """Audit safe regular members and contents of one Cargo ``.crate``."""
    root = repository.resolve()
    identity = package_identity(repository)
    expected_path = root / "target" / "package" / f"{identity.archive_root}.crate"
    if crate.resolve() != expected_path:
        raise AuditFailure("package artifact is not the Cargo package output")
    if crate.suffix != ".crate":
        raise AuditFailure("package artifact must have a .crate suffix")
    if crate.name != f"{identity.archive_root}.crate":
        raise AuditFailure("package artifact filename does not match Cargo name and version")
    if not crate.is_file() or crate.is_symlink():
        raise AuditFailure("package artifact is not a regular file")
    markers = residue_markers(repository)
    member_count = 0
    total_bytes = 0
    roots: set[str] = set()
    names: set[str] = set()
    packaged_manifest: bytes | None = None
    original_manifest_seen = False
    try:
        archive = tarfile.open(crate, mode="r:gz")
    except (tarfile.TarError, OSError) as error:
        raise AuditFailure("cannot open package archive") from error
    with archive:
        for member in archive:
            member_count += 1
            if member_count > MAX_ARCHIVE_MEMBERS:
                raise AuditFailure("package archive has too many members")
            path = _safe_relative_path(member.name, "package member")
            if path.parts[0] != identity.archive_root:
                raise AuditFailure("package archive root does not match Cargo name and version")
            if member.name in names:
                raise AuditFailure(f"package archive repeats member: {member.name}")
            names.add(member.name)
            roots.add(path.parts[0])
            _audit_generated_path(path, "package member")
            _scan_name(member.name, f"package member {member.name!r}", markers)
            if member.isdir():
                continue
            if not member.isfile():
                raise AuditFailure(f"package member is not a regular file: {member.name}")
            if member.size < 0 or member.size > MAX_ARCHIVE_FILE_BYTES:
                raise AuditFailure(f"package member exceeds audit bound: {member.name}")
            total_bytes += member.size
            if total_bytes > MAX_ARCHIVE_TOTAL_BYTES:
                raise AuditFailure("package archive expands beyond audit bound")
            stream = archive.extractfile(member)
            if stream is None:
                raise AuditFailure(f"cannot read package member: {member.name}")
            relative = PurePosixPath(*path.parts[1:])
            if not relative.parts:
                raise AuditFailure(f"package root is a regular file: {member.name}")
            with stream:
                if relative == PurePosixPath("Cargo.toml"):
                    if member.size > MAX_MANIFEST_BYTES:
                        raise AuditFailure("packaged Cargo.toml exceeds audit bound")
                    packaged_manifest = stream.read(MAX_MANIFEST_BYTES + 1)
                    if len(packaged_manifest) != member.size:
                        raise AuditFailure("packaged Cargo.toml size is inconsistent")
                    _scan_bytes(
                        packaged_manifest,
                        f"package member {member.name}",
                        markers,
                    )
                elif relative == PurePosixPath("Cargo.toml.orig"):
                    original_manifest_seen = True
                    with (root / "Cargo.toml").open("rb") as expected:
                        _scan_stream(
                            stream,
                            f"package member {member.name}",
                            markers,
                            expected=expected,
                        )
                elif relative.as_posix() in CARGO_GENERATED_MEMBERS:
                    _scan_stream(stream, f"package member {member.name}", markers)
                else:
                    source = root.joinpath(*relative.parts)
                    try:
                        metadata = source.lstat()
                    except OSError as error:
                        raise AuditFailure(
                            f"package member has no worktree source: {member.name}"
                        ) from error
                    if not stat.S_ISREG(metadata.st_mode) or source.is_symlink():
                        raise AuditFailure(
                            f"package member source is not a regular file: {member.name}"
                        )
                    with source.open("rb") as expected:
                        _scan_stream(
                            stream,
                            f"package member {member.name}",
                            markers,
                            expected=expected,
                        )
    if member_count == 0:
        raise AuditFailure("package archive is empty")
    if roots != {identity.archive_root}:
        raise AuditFailure("package archive root does not match Cargo name and version")
    if packaged_manifest is None:
        raise AuditFailure("package archive is missing Cargo.toml")
    if not original_manifest_seen:
        raise AuditFailure("package archive is missing Cargo.toml.orig")
    if _package_identity_from_manifest(packaged_manifest) != identity:
        raise AuditFailure("packaged Cargo.toml identity does not match repository")
    return member_count


def audit_binary(binary: Path, repository: Path) -> None:
    """Audit strings in one executable regular file."""
    identity = package_identity(repository)
    expected_path = (
        repository.resolve()
        / "target"
        / APPLE_SILICON_TARGET
        / "release"
        / identity.name
    )
    if binary.resolve() != expected_path:
        raise AuditFailure("release binary path does not match Cargo release output")
    metadata = binary.lstat()
    if not stat.S_ISREG(metadata.st_mode) or binary.is_symlink():
        raise AuditFailure("release binary is not a regular file")
    if not os.access(binary, os.X_OK):
        raise AuditFailure("release binary is not executable")
    with binary.open("rb") as stream:
        _scan_stream(stream, "release binary", residue_markers(repository))
    _audit_macho_release_contract(binary)
    try:
        result = subprocess.run(
            [os.fspath(binary.resolve()), "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            env={"LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AuditFailure("release binary version probe failed") from error
    expected = f"{identity.name} {identity.version}\n".encode()
    if result.returncode != 0 or result.stdout != expected or result.stderr:
        raise AuditFailure("release binary version does not match Cargo name and version")


def _audit_macho_release_contract(binary: Path) -> None:
    with binary.open("rb") as stream:
        header = stream.read(MACHO_HEADER_BYTES)
        if len(header) != MACHO_HEADER_BYTES:
            raise AuditFailure("release binary has no complete Mach-O header")
        magic = int.from_bytes(header[0:4], "little")
        cpu_type = int.from_bytes(header[4:8], "little")
        file_type = int.from_bytes(header[12:16], "little")
        command_count = int.from_bytes(header[16:20], "little")
        command_bytes = int.from_bytes(header[20:24], "little")
        if magic != 0xFEEDFACF or cpu_type != 0x0100000C or file_type != 2:
            raise AuditFailure("release binary is not a thin arm64 Mach-O executable")
        if (
            command_count == 0
            or command_count > MAX_MACHO_LOAD_COMMANDS
            or command_bytes > MAX_MACHO_LOAD_COMMAND_BYTES
        ):
            raise AuditFailure("release binary has invalid Mach-O load-command bounds")
        commands = stream.read(command_bytes)
        if len(commands) != command_bytes:
            raise AuditFailure("release binary has truncated Mach-O load commands")

    minimum_versions: list[tuple[int, int, int]] = []
    offset = 0
    for _ in range(command_count):
        if offset + 8 > len(commands):
            raise AuditFailure("release binary has truncated Mach-O load-command header")
        command = int.from_bytes(commands[offset : offset + 4], "little")
        size = int.from_bytes(commands[offset + 4 : offset + 8], "little")
        if size < 8 or size % 8 != 0 or offset + size > len(commands):
            raise AuditFailure("release binary has invalid Mach-O load-command size")
        if command == LC_BUILD_VERSION:
            if size < 24:
                raise AuditFailure("release binary has truncated LC_BUILD_VERSION")
            platform = int.from_bytes(commands[offset + 8 : offset + 12], "little")
            if platform != PLATFORM_MACOS:
                raise AuditFailure("release binary does not target the macOS Mach-O platform")
            packed = int.from_bytes(commands[offset + 12 : offset + 16], "little")
            minimum_versions.append(_unpack_apple_version(packed))
        elif command == LC_VERSION_MIN_MACOSX:
            if size < 16:
                raise AuditFailure("release binary has truncated LC_VERSION_MIN_MACOSX")
            packed = int.from_bytes(commands[offset + 8 : offset + 12], "little")
            minimum_versions.append(_unpack_apple_version(packed))
        elif command in DYLIB_LOAD_COMMANDS:
            _audit_macho_dylib_command(commands, offset, size)
        offset += size
    if offset != len(commands):
        raise AuditFailure("release binary Mach-O load-command size is inconsistent")
    if not minimum_versions:
        raise AuditFailure("release binary has no macOS minimum-version load command")
    if any(version != RELEASE_MINIMUM_MACOS for version in minimum_versions):
        expected = ".".join(map(str, RELEASE_MINIMUM_MACOS[:2]))
        actual = ", ".join(".".join(map(str, version)) for version in minimum_versions)
        raise AuditFailure(
            f"release binary minimum macOS version is {actual}; expected {expected}"
        )


def _audit_macho_dylib_command(commands: bytes, offset: int, size: int) -> None:
    if size < 24:
        raise AuditFailure("release binary has truncated Mach-O dylib command")
    name_offset = int.from_bytes(commands[offset + 8 : offset + 12], "little")
    if name_offset < 24 or name_offset >= size:
        raise AuditFailure("release binary has invalid Mach-O dylib name offset")
    encoded = commands[offset + name_offset : offset + size]
    terminator = encoded.find(b"\0")
    if terminator <= 0:
        raise AuditFailure("release binary has unterminated Mach-O dylib name")
    try:
        dependency = encoded[:terminator].decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditFailure("release binary has non-UTF-8 Mach-O dylib name") from error
    dependency_path = PurePosixPath(dependency)
    if (
        not dependency_path.is_absolute()
        or dependency_path.as_posix() != dependency
        or any(part in {".", ".."} for part in dependency_path.parts)
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in dependency)
    ):
        raise AuditFailure("release binary has invalid Mach-O dylib name")
    if not dependency.startswith(SYSTEM_DYLIB_PREFIXES):
        raise AuditFailure(
            f"release binary has non-system dynamic dependency: {dependency}"
        )


def _unpack_apple_version(value: int) -> tuple[int, int, int]:
    return ((value >> 16) & 0xFFFF, (value >> 8) & 0xFF, value & 0xFF)


def run(
    repository: Path,
    crate: Path | None = None,
    binary: Path | None = None,
) -> tuple[int, int]:
    """Run all release audits and return tracked/member counts."""
    root = repository.resolve()
    identity = package_identity(root)
    if crate is None:
        crate = root / "target" / "package" / f"{identity.archive_root}.crate"
    if binary is None:
        binary = root / "target" / APPLE_SILICON_TARGET / "release" / identity.name
    tracked = audit_repository(repository)
    members = audit_crate(crate, repository)
    audit_binary(binary, repository)
    return tracked, members


def package_identity(repository: Path) -> PackageIdentity:
    """Read the simple string package identity from the repository manifest."""
    manifest = repository.resolve() / "Cargo.toml"
    try:
        metadata = manifest.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or manifest.is_symlink()
            or metadata.st_size > MAX_MANIFEST_BYTES
        ):
            raise AuditFailure("repository Cargo.toml is not a bounded regular file")
        data = manifest.read_bytes()
    except OSError as error:
        raise AuditFailure("cannot read repository Cargo.toml") from error
    identity = _package_identity_from_manifest(data)
    if identity.name != "emelex":
        raise AuditFailure("repository package name is not emelex")
    return identity


def _package_identity_from_manifest(data: bytes) -> PackageIdentity:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditFailure("Cargo.toml is not UTF-8") from error
    section: str | None = None
    fields: dict[str, str] = {}
    assignment = re.compile(r'^([A-Za-z0-9_-]+)\s*=\s*"([^"\\]*)"\s*$')
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line
            continue
        if section != "[package]":
            continue
        match = assignment.fullmatch(line)
        if match and match.group(1) in {"name", "version"}:
            key, value = match.groups()
            if key in fields:
                raise AuditFailure(f"Cargo.toml repeats package.{key}")
            fields[key] = value
    if set(fields) != {"name", "version"}:
        raise AuditFailure("Cargo.toml has no unambiguous package name and version")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", fields["name"]):
        raise AuditFailure("Cargo package name is invalid")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", fields["version"]):
        raise AuditFailure("Cargo package version is invalid")
    return PackageIdentity(fields["name"], fields["version"])


def _audit_workspace_generated_residue(root: Path) -> None:
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative = current_path.relative_to(root)
        if relative == Path("."):
            directories[:] = sorted(
                name for name in directories if name not in IGNORED_WORKSPACE_ROOTS
            )
        else:
            directories.sort()
        files.sort()
        if relative == Path(".") and ".git" in files:
            files.remove(".git")
        if relative == Path("agents") and "worktrees" in directories:
            worktrees = current_path / "worktrees"
            entries = sorted(worktrees.iterdir(), key=lambda entry: entry.name)
            material = [entry.name for entry in entries if entry.name != ".gitkeep"]
            if material:
                raise AuditFailure("workspace contains harness worktree material")
            sentinel = worktrees / ".gitkeep"
            if sentinel.exists() or sentinel.is_symlink():
                metadata = sentinel.lstat()
                if not stat.S_ISREG(metadata.st_mode) or sentinel.is_symlink():
                    raise AuditFailure("worktree sentinel is not a regular file")
            directories.remove("worktrees")
        for directory in directories:
            path = relative / directory
            _audit_generated_path(path, "workspace path")
        for filename in files:
            _audit_generated_path(relative / filename, "workspace path")


def _audit_generated_path(
    path: PurePosixPath | Path,
    source: str,
    *,
    allow_worktree_sentinel: bool = False,
) -> None:
    if any(part in FORBIDDEN_GENERATED_COMPONENTS for part in path.parts):
        raise AuditFailure(f"{source} contains generated directory residue")
    if path.suffix.lower() == ".pyc":
        raise AuditFailure(f"{source} contains Python bytecode residue")
    if path.name in FORBIDDEN_GENERATED_NAMES or path.suffix.lower() == ".profraw":
        raise AuditFailure(f"{source} contains generated file residue")
    sentinel = path.parts == ("agents", "worktrees", ".gitkeep")
    has_worktree_pair = any(
        path.parts[index : index + 2] == ("agents", "worktrees")
        for index in range(len(path.parts) - 1)
    )
    if has_worktree_pair and not (
        allow_worktree_sentinel and sentinel
    ):
        raise AuditFailure(f"{source} contains harness worktree material")


def _safe_relative_path(value: str, source: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AuditFailure(f"{source} is not a safe relative path")
    return path


def _scan_name(value: str, location: str, markers: Iterable[Marker]) -> None:
    encoded = value.encode("utf-8").lower()
    for marker in markers:
        if marker.value in encoded:
            raise AuditFailure(f"{location} contains {marker.label}")


def _scan_stream(
    stream: BinaryIO,
    location: str,
    markers: Iterable[Marker],
    *,
    expected: BinaryIO | None = None,
) -> None:
    selected = tuple(markers)
    overlap = max((len(marker.value) for marker in selected), default=1) - 1
    tail = b""
    while True:
        chunk = stream.read(READ_SIZE)
        if not chunk:
            if expected is not None and expected.read(1):
                raise AuditFailure(f"{location} does not match frozen worktree source")
            return
        candidate = (tail + chunk).lower()
        for marker in selected:
            if marker.value in candidate:
                raise AuditFailure(f"{location} contains {marker.label}")
        if expected is not None and expected.read(len(chunk)) != chunk:
            raise AuditFailure(f"{location} does not match frozen worktree source")
        tail = candidate[-overlap:] if overlap else b""


def _scan_bytes(data: bytes, location: str, markers: Iterable[Marker]) -> None:
    lowered = data.lower()
    for marker in markers:
        if marker.value in lowered:
            raise AuditFailure(f"{location} contains {marker.label}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="Git checkout whose tracked files are audited (default: current directory)",
    )
    parser.add_argument(
        "--crate",
        type=Path,
        help="Cargo .crate artifact (default: derived Cargo package output)",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="release executable (default: derived Cargo release output)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tracked, members = run(args.repository, args.crate, args.binary)
    except (AuditFailure, OSError) as error:
        print(f"release audit: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"release audit: PASS ({tracked} tracked files, "
        f"{members} package members, executable scanned)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
