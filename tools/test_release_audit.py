#!/usr/bin/env python3
"""Regression tests for the deterministic release residue audit."""

from __future__ import annotations

import io
import os
import struct
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from release_audit import (
    APPLE_SILICON_TARGET,
    DYLIB_LOAD_COMMANDS,
    LC_LOAD_DYLIB,
    MAX_MANIFEST_BYTES,
    AuditFailure,
    _audit_workspace_generated_residue,
    audit_binary,
    audit_crate,
    audit_repository,
    run as run_release_audit,
)

FORMER_NAME = (b"NiT" + b"PiN").decode()
PRIVATE_PATH = (b"/" + b"Users" + b"/example/private").decode()
EARLY_RELEASE_OVERRIDES = (
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_TARGET_DIR",
    "CARGO_BUILD_TARGET",
    "RUSTUP_TOOLCHAIN",
    "RUSTC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
)
NATIVE_BUILD_OVERRIDES = (
    "MACOSX_DEPLOYMENT_TARGET",
    "CC",
    "CXX",
    "CPP",
    "AR",
    "RANLIB",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "HOST_CC",
    "HOST_CXX",
    "HOST_AR",
    "HOST_RANLIB",
    "HOST_CFLAGS",
    "HOST_CXXFLAGS",
    "HOST_CPPFLAGS",
    "HOST_LDFLAGS",
    "TARGET_CC",
    "TARGET_CXX",
    "TARGET_AR",
    "TARGET_RANLIB",
    "TARGET_CFLAGS",
    "TARGET_CXXFLAGS",
    "TARGET_CPPFLAGS",
    "TARGET_LDFLAGS",
    "CC_aarch64-apple-darwin",
    "CXX_aarch64-apple-darwin",
    "AR_aarch64-apple-darwin",
    "RANLIB_aarch64-apple-darwin",
    "CFLAGS_aarch64-apple-darwin",
    "CXXFLAGS_aarch64-apple-darwin",
    "CPPFLAGS_aarch64-apple-darwin",
    "LDFLAGS_aarch64-apple-darwin",
    "CC_aarch64_apple_darwin",
    "CXX_aarch64_apple_darwin",
    "AR_aarch64_apple_darwin",
    "RANLIB_aarch64_apple_darwin",
    "CFLAGS_aarch64_apple_darwin",
    "CXXFLAGS_aarch64_apple_darwin",
    "CPPFLAGS_aarch64_apple_darwin",
    "LDFLAGS_aarch64_apple_darwin",
    "LIBCLANG_PATH",
    "CLANG_PATH",
    "LLVM_CONFIG_PATH",
    "LD_LIBRARY_PATH",
    "BINDGEN_EXTRA_CLANG_ARGS",
    "BINDGEN_EXTRA_CLANG_ARGS_aarch64-apple-darwin",
    "BINDGEN_EXTRA_CLANG_ARGS_aarch64_apple_darwin",
    "DEVELOPER_DIR",
    "TOOLCHAINS",
    "SDKROOT",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "CLANG_MODULE_CACHE_PATH",
    "CRATE_CC_NO_DEFAULTS",
    "CMAKE",
    "TARGET_CMAKE",
    "CMAKE_aarch64-apple-darwin",
    "CMAKE_aarch64_apple_darwin",
    "CMAKE_GENERATOR",
    "TARGET_CMAKE_GENERATOR",
    "CMAKE_GENERATOR_aarch64-apple-darwin",
    "CMAKE_GENERATOR_aarch64_apple_darwin",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_TOOLCHAIN_FILE",
    "TARGET_CMAKE_TOOLCHAIN_FILE",
    "CMAKE_TOOLCHAIN_FILE_aarch64-apple-darwin",
    "CMAKE_TOOLCHAIN_FILE_aarch64_apple_darwin",
    "CMAKE_PREFIX_PATH",
    "TARGET_CMAKE_PREFIX_PATH",
    "CMAKE_PREFIX_PATH_aarch64-apple-darwin",
    "CMAKE_PREFIX_PATH_aarch64_apple_darwin",
    "CMAKE_OSX_ARCHITECTURES",
    "CMAKE_OSX_DEPLOYMENT_TARGET",
    "CMAKE_OSX_SYSROOT",
    "AWS_LC_SYS_USE_SYSTEM",
    "AWS_LC_SYS_USE_SYSTEM_aarch64_apple_darwin",
    "LIBSQLITE3_SYS_USE_PKG_CONFIG",
    "SQLITE_MAX_VARIABLE_NUMBER",
    "SQLITE_MAX_EXPR_DEPTH",
    "SQLITE_MAX_COLUMN",
    "LIBSQLITE3_FLAGS",
    "ZSTD_SYS_USE_PKG_CONFIG",
    "RUSTONIG_SYSTEM_LIBONIG",
    "RUSTONIG_DYNAMIC_LIBONIG",
    "RUSTONIG_STATIC_LIBONIG",
    "DOCS_RS",
)


def initialize_repository(root: Path) -> None:
    subprocess.run(["git", "init", "-q", root], check=True)
    (root / "README.md").write_text("standalone local inference\n")
    (root / "Cargo.toml").write_text(
        '[package]\nname = "emelex"\nversion = "1.0.0"\nedition = "2024"\n'
    )
    subprocess.run(["git", "-C", root, "add", "Cargo.toml", "README.md"], check=True)


def write_crate(path: Path, members: dict[str, bytes]) -> None:
    original_manifest = (
        b'[package]\nname = "emelex"\nversion = "1.0.0"\nedition = "2024"\n'
    )
    selected = {
        "emelex-1.0.0/Cargo.toml": original_manifest,
        "emelex-1.0.0/Cargo.toml.orig": original_manifest,
        **members,
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, content in selected.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))


def write_binary(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755)


def release_binary(root: Path) -> Path:
    return root / "target" / APPLE_SILICON_TARGET / "release" / "emelex"


def macho_executable(
    extra: bytes = b"",
    minimum_macos: tuple[int, int, int] = (26, 5, 0),
    load_commands: tuple[bytes, ...] = (),
) -> bytes:
    packed_version = (
        minimum_macos[0] << 16 | minimum_macos[1] << 8 | minimum_macos[2]
    )
    build_version = struct.pack(
        "<IIIIII",
        0x32,
        24,
        1,
        packed_version,
        packed_version,
        0,
    )
    commands = build_version + b"".join(load_commands)
    header = struct.pack(
        "<IIIIIIII",
        0xFEEDFACF,
        0x0100000C,
        0,
        2,
        1 + len(load_commands),
        len(commands),
        0,
        0,
    )
    return header + commands + extra


def dylib_command(path: bytes, command: int = LC_LOAD_DYLIB) -> bytes:
    encoded = path + b"\0"
    size = (24 + len(encoded) + 7) & ~7
    return (
        struct.pack("<IIIIII", command, size, 24, 0, 0, 0)
        + encoded
        + b"\0" * (size - 24 - len(encoded))
    )


def version_result(version: str = "1.0.0") -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=f"emelex {version}\n".encode(),
        stderr=b"",
    )


class ReleaseAuditTests(unittest.TestCase):
    def test_safe_repository_archive_and_binary_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            package = root / "target" / "package" / "emelex-1.0.0.crate"
            binary = release_binary(root)
            package.parent.mkdir(parents=True)
            write_crate(
                package,
                {"emelex-1.0.0/README.md": b"standalone local inference\n"},
            )
            write_binary(binary, macho_executable())
            self.assertEqual(audit_repository(root), 2)
            self.assertEqual(audit_crate(package, root), 3)
            with patch("release_audit.subprocess.run", return_value=version_result()):
                audit_binary(binary, root)
            system_run = subprocess.run

            def dispatch(command: list[str], **kwargs: object) -> object:
                if command[-1] == "--version":
                    return version_result()
                return system_run(command, **kwargs)

            with patch("release_audit.subprocess.run", side_effect=dispatch):
                self.assertEqual(run_release_audit(root), (2, 3))

    def test_tracked_name_and_content_residue_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            bad_name = root / f"{FORMER_NAME}.txt"
            bad_name.write_text("safe")
            subprocess.run(["git", "-C", root, "add", bad_name.name], check=True)
            with self.assertRaisesRegex(AuditFailure, "former product"):
                audit_repository(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            (root / "README.md").write_text(f"legacy {FORMER_NAME}\n")
            with self.assertRaisesRegex(AuditFailure, "former product"):
                audit_repository(root)

    def test_private_path_and_generated_residue_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            (root / "README.md").write_text(PRIVATE_PATH)
            with self.assertRaisesRegex(AuditFailure, "private macOS home"):
                audit_repository(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            cache = root / "tools" / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "tool.pyc").write_bytes(b"generated")
            with self.assertRaisesRegex(AuditFailure, "generated directory"):
                audit_repository(root)

    def test_archive_member_name_and_content_residue_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            package = root / "target" / "package" / "emelex-1.0.0.crate"
            package.parent.mkdir(parents=True)
            write_crate(package, {f"emelex-1.0.0/{FORMER_NAME}.md": b"safe"})
            with self.assertRaisesRegex(AuditFailure, "former product"):
                audit_crate(package, root)

            write_crate(
                package,
                {"emelex-1.0.0/README.md": f"legacy {FORMER_NAME}".encode()},
            )
            with self.assertRaisesRegex(AuditFailure, "former product"):
                audit_crate(package, root)

            write_crate(
                package,
                {"emelex-1.0.0/README.md": PRIVATE_PATH.encode()},
            )
            with self.assertRaisesRegex(AuditFailure, "private macOS home"):
                audit_crate(package, root)

    def test_binary_string_residue_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            binary = release_binary(root)
            write_binary(binary, macho_executable(PRIVATE_PATH.encode()))
            with self.assertRaisesRegex(AuditFailure, "private macOS home"):
                audit_binary(binary, root)

            write_binary(binary, macho_executable(FORMER_NAME.encode()))
            with self.assertRaisesRegex(AuditFailure, "former product"):
                audit_binary(binary, root)

    def test_binary_minimum_macos_contract_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            binary = release_binary(root)
            write_binary(binary, macho_executable(minimum_macos=(27, 0, 0)))
            with self.assertRaisesRegex(AuditFailure, "minimum macOS version"):
                audit_binary(binary, root)

            missing_command = struct.pack(
                "<IIIIIIII",
                0xFEEDFACF,
                0x0100000C,
                0,
                2,
                0,
                0,
                0,
                0,
            )
            write_binary(binary, missing_command)
            with self.assertRaisesRegex(AuditFailure, "load-command bounds"):
                audit_binary(binary, root)

    def test_binary_dynamic_dependencies_are_system_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            binary = release_binary(root)
            write_binary(
                binary,
                macho_executable(
                    load_commands=(dylib_command(b"/usr/lib/libSystem.B.dylib"),)
                ),
            )
            with patch("release_audit.subprocess.run", return_value=version_result()):
                audit_binary(binary, root)

            for command in DYLIB_LOAD_COMMANDS:
                write_binary(
                    binary,
                    macho_executable(
                        load_commands=(
                            dylib_command(
                                b"/opt/homebrew/lib/libonig.dylib",
                                command,
                            ),
                        )
                    ),
                )
                with self.assertRaisesRegex(AuditFailure, "non-system dynamic"):
                    audit_binary(binary, root)
                write_binary(
                    binary,
                    macho_executable(
                        load_commands=(
                            dylib_command(
                                b"/usr/lib/../../opt/homebrew/libevil.dylib",
                                command,
                            ),
                        )
                    ),
                )
                with self.assertRaisesRegex(AuditFailure, "invalid Mach-O dylib"):
                    audit_binary(binary, root)

    def test_binary_rejects_malformed_dylib_load_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            binary = release_binary(root)
            malformed_commands = (
                struct.pack("<II", LC_LOAD_DYLIB, 12) + b"\0" * 4,
                struct.pack("<IIIIII", LC_LOAD_DYLIB, 24, 24, 0, 0, 0),
                struct.pack("<IIIIII", LC_LOAD_DYLIB, 32, 24, 0, 0, 0)
                + b"12345678",
                struct.pack("<IIIIII", LC_LOAD_DYLIB, 32, 24, 0, 0, 0)
                + b"\xff\0"
                + b"\0" * 6,
            )
            expected_errors = (
                "load-command size",
                "dylib name offset",
                "unterminated",
                "non-UTF-8",
            )
            for command, expected in zip(
                malformed_commands,
                expected_errors,
                strict=True,
            ):
                write_binary(
                    binary,
                    macho_executable(load_commands=(command,)),
                )
                with self.assertRaisesRegex(AuditFailure, expected):
                    audit_binary(binary, root)

    def test_cross_chunk_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            binary = release_binary(root)
            marker = FORMER_NAME.encode()
            header = macho_executable()
            content = header + b"x" * ((1 << 20) - 2 - len(header)) + marker
            write_binary(binary, content)
            with self.assertRaisesRegex(AuditFailure, "former product"):
                audit_binary(binary, root)

    def test_generated_and_unsafe_archive_members_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            package = root / "target" / "package" / "emelex-1.0.0.crate"
            package.parent.mkdir(parents=True)
            write_crate(package, {"emelex-1.0.0/__pycache__/tool.pyc": b"cache"})
            with self.assertRaisesRegex(AuditFailure, "generated directory"):
                audit_crate(package, root)

            write_crate(
                package,
                {"emelex-1.0.0/agents/worktrees/leak/file": b"material"},
            )
            with self.assertRaisesRegex(AuditFailure, "worktree material"):
                audit_crate(package, root)

            write_crate(package, {"../escape": b"unsafe"})
            with self.assertRaisesRegex(AuditFailure, "safe relative path"):
                audit_crate(package, root)

            with tarfile.open(package, "w:gz") as archive:
                link = tarfile.TarInfo("emelex-1.0.0/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "README.md"
                archive.addfile(link)
            with self.assertRaisesRegex(AuditFailure, "not a regular file"):
                audit_crate(package, root)

            oversized = b"x" * (MAX_MANIFEST_BYTES + 1)
            write_crate(
                package,
                {"emelex-1.0.0/Cargo.toml": oversized},
            )
            with self.assertRaisesRegex(AuditFailure, "Cargo.toml exceeds"):
                audit_crate(package, root)

    def test_auditor_source_does_not_contain_its_own_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            source = Path(__file__).with_name("release_audit.py")
            destination = root / "release_audit.py"
            destination.write_bytes(source.read_bytes())
            subprocess.run(["git", "-C", root, "add", destination.name], check=True)
            self.assertEqual(audit_repository(root), 3)

    def test_artifact_identity_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            package = root / "target" / "package" / "wrong-1.0.0.crate"
            package.parent.mkdir(parents=True)
            write_crate(package, {"emelex-1.0.0/README.md": b"safe"})
            with self.assertRaisesRegex(AuditFailure, "Cargo package output"):
                audit_crate(package, root)

            binary = release_binary(root)
            write_binary(binary, macho_executable())
            with (
                patch(
                    "release_audit.subprocess.run",
                    return_value=version_result("9.9.9"),
                ),
                self.assertRaisesRegex(AuditFailure, "version"),
            ):
                audit_binary(binary, root)

            stale_binary = root / "target" / "release" / "emelex"
            write_binary(stale_binary, macho_executable())
            with self.assertRaisesRegex(AuditFailure, "release output"):
                audit_binary(stale_binary, root)

    def test_stale_package_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            package = root / "target" / "package" / "emelex-1.0.0.crate"
            package.parent.mkdir(parents=True)
            write_crate(package, {"emelex-1.0.0/README.md": b"stale\n"})
            with self.assertRaisesRegex(AuditFailure, "frozen worktree"):
                audit_crate(package, root)

    def test_release_gate_builds_every_artifact_in_sanitized_environment(self) -> None:
        gate = Path(__file__).with_name("release_gate.sh").read_text()
        for name in NATIVE_BUILD_OVERRIDES:
            self.assertIn(f"\t{name}\n", gate)
        native_guard = gate.index('for override_name in "${native_build_overrides[@]}"')
        deployment_target = gate.index('release_deployment_target="26.5"')
        sanitizer = gate.index("release_cargo_environment=(")
        clean = gate.index(
            'release_cargo clean --manifest-path "$repository/Cargo.toml" '
            '--target-dir "$repository/target"'
        )
        package = gate.index("release_cargo package ")
        build = gate.index("release_cargo build ")
        audit = gate.index("python3 tools/release_audit.py")
        self.assertLess(native_guard, deployment_target)
        self.assertLess(deployment_target, sanitizer)
        self.assertLess(sanitizer, clean)
        self.assertLess(clean, package)
        self.assertLess(package, build)
        self.assertLess(build, audit)
        self.assertNotIn("release_cargo clean -p", gate)
        self.assertNotIn("release_cargo clean --release", gate)
        self.assertIn("/usr/bin/env -i", gate)
        self.assertIn('"CARGO_HOME=$isolated_cargo_home"', gate)
        self.assertIn('"AWS_LC_SYS_USE_SYSTEM=0"', gate)
        self.assertIn('cd "$release_driver"', gate)

    def test_release_gate_rejects_representative_native_overrides(self) -> None:
        gate = Path(__file__).with_name("release_gate.sh")
        clean_environment = os.environ.copy()
        for name in EARLY_RELEASE_OVERRIDES + NATIVE_BUILD_OVERRIDES:
            clean_environment.pop(name, None)
        for name in (
            "MACOSX_DEPLOYMENT_TARGET",
            "CXXFLAGS",
            "CC_aarch64-apple-darwin",
            "BINDGEN_EXTRA_CLANG_ARGS",
            "BINDGEN_EXTRA_CLANG_ARGS_aarch64-apple-darwin",
            "CLANG_PATH",
            "LLVM_CONFIG_PATH",
            "LD_LIBRARY_PATH",
            "AWS_LC_SYS_USE_SYSTEM",
            "AWS_LC_SYS_USE_SYSTEM_aarch64_apple_darwin",
            "LIBSQLITE3_SYS_USE_PKG_CONFIG",
            "SQLITE_MAX_VARIABLE_NUMBER",
            "SQLITE_MAX_EXPR_DEPTH",
            "SQLITE_MAX_COLUMN",
            "LIBSQLITE3_FLAGS",
            "ZSTD_SYS_USE_PKG_CONFIG",
            "RUSTONIG_SYSTEM_LIBONIG",
            "RUSTONIG_DYNAMIC_LIBONIG",
            "RUSTONIG_STATIC_LIBONIG",
            "DOCS_RS",
            "CRATE_CC_NO_DEFAULTS",
            "CMAKE_TOOLCHAIN_FILE_aarch64-apple-darwin",
            "TARGET_CMAKE_GENERATOR",
        ):
            environment = clean_environment | {name: "unexpected"}
            completed = subprocess.run(
                ["bash", gate, "--invalid-test-option"],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(f"ambient native build override {name}", completed.stderr)

    def test_script_binary_and_gitfile_edge_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_repository(root)
            binary = release_binary(root)
            write_binary(binary, b"#!/bin/sh\nprintf 'emelex 1.0.0\\n'\n")
            with self.assertRaisesRegex(AuditFailure, "Mach-O"):
                audit_binary(binary, root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").write_text("gitdir: elsewhere\n")
            _audit_workspace_generated_residue(root)


if __name__ == "__main__":
    unittest.main()
