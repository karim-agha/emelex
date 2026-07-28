#!/usr/bin/env python3
"""Generate the locked Rust dependency license bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPOSITORY_ROOT / "licenses" / "RUST-DEPENDENCIES.md"
LICENSE_PREFIXES = ("license", "copying", "notice", "copyright")
RUST_RELEASE = "1.97.0"
RUST_COMMIT = "2d8144b7880597b6e6d3dfd63a9a9efae3f533d3"
COMPILER_BUILTINS_LICENSE_SHA256 = (
    "ab6eec6caf0fa5775e411c7a8bc6a45c4ef2956b0980b157ab74fc5cd62a928b"
)


@dataclass(frozen=True)
class Package:
    """Dependency metadata needed by the generated bundle."""

    name: str
    version: str
    source: str
    declared_license: str
    authors: tuple[str, ...]
    license_documents: tuple[str, ...]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the checked-in bundle differs from Cargo.lock",
    )
    return parser.parse_args()


def cargo_metadata() -> dict[str, object]:
    completed = subprocess.run(
        [
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--offline",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "cargo metadata failed")
    return json.loads(completed.stdout)


def candidate_license_files(package: dict[str, object]) -> list[Path]:
    manifest = Path(require_text(package, "manifest_path"))
    crate_root = manifest.parent
    declared_file = package.get("license_file")
    if isinstance(declared_file, str):
        path = crate_root / declared_file
        if not path.is_file():
            raise RuntimeError(f"{package_label(package)} license_file is absent: {path}")
        return [path]

    candidates = [
        path
        for path in crate_root.iterdir()
        if path.is_file() and path.name.casefold().startswith(LICENSE_PREFIXES)
    ]
    return sorted(candidates, key=lambda path: path.name.casefold())


def require_text(package: dict[str, object], key: str) -> str:
    value = package.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{package_label(package)} has no {key}")
    return value


def package_label(package: dict[str, object]) -> str:
    return f"{package.get('name', '?')} {package.get('version', '?')}"


def normalize_license(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def compiler_builtins() -> tuple[Package, str, str]:
    """Resolve and verify the compiler runtime embedded by the pinned rustc."""
    version = subprocess.run(
        ["rustc", "-Vv"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if version.returncode != 0:
        raise RuntimeError(version.stderr.strip() or "rustc -Vv failed")
    fields = dict(
        line.split(":", 1)
        for line in version.stdout.splitlines()
        if ":" in line
    )
    release = fields.get("release", "").strip()
    commit = fields.get("commit-hash", "").strip()
    host = fields.get("host", "").strip()
    if release != RUST_RELEASE or commit != RUST_COMMIT:
        raise RuntimeError(
            "Rust toolchain drifted: expected "
            f"{RUST_RELEASE} ({RUST_COMMIT}), got {release} ({commit})"
        )
    if host != "aarch64-apple-darwin":
        raise RuntimeError(f"unsupported Rust host for release provenance: {host}")

    sysroot_result = subprocess.run(
        ["rustc", "--print", "sysroot"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if sysroot_result.returncode != 0:
        raise RuntimeError(sysroot_result.stderr.strip() or "rustc sysroot query failed")
    sysroot = Path(sysroot_result.stdout.strip())
    runtime_dir = sysroot / "lib" / "rustlib" / host / "lib"
    runtime_archives = sorted(runtime_dir.glob("libcompiler_builtins-*.rlib"))
    if len(runtime_archives) != 1:
        raise RuntimeError(
            "expected exactly one pinned compiler_builtins runtime archive, found "
            f"{len(runtime_archives)} under {runtime_dir}"
        )

    license_path = (
        sysroot
        / "lib"
        / "rustlib"
        / "src"
        / "rust"
        / "library"
        / "compiler-builtins"
        / "LICENSE.txt"
    )
    license_text = normalize_license(license_path)
    digest = hashlib.sha256(license_text.encode()).hexdigest()
    if digest != COMPILER_BUILTINS_LICENSE_SHA256:
        raise RuntimeError(
            "pinned compiler_builtins license drifted: expected "
            f"{COMPILER_BUILTINS_LICENSE_SHA256}, got {digest}"
        )
    package = Package(
        name="compiler_builtins (rustc runtime)",
        version=release,
        source=f"rustc {commit}; {runtime_archives[0].name}",
        declared_license="MIT AND Apache-2.0 WITH LLVM-exception",
        authors=("The Rust Project Developers",),
        license_documents=(digest[:16],),
    )
    return package, digest[:16], license_text


def dependency_packages(
    metadata: dict[str, object],
) -> tuple[list[Package], dict[str, str], dict[str, set[str]]]:
    workspace_members = set(metadata.get("workspace_members", []))
    packages = metadata.get("packages")
    if not isinstance(packages, list):
        raise RuntimeError("cargo metadata returned no packages")

    records: list[Package] = []
    documents: dict[str, str] = {}
    document_users: dict[str, set[str]] = defaultdict(set)
    for raw_package in packages:
        if not isinstance(raw_package, dict) or raw_package.get("id") in workspace_members:
            continue
        name = require_text(raw_package, "name")
        version = require_text(raw_package, "version")
        source = require_text(raw_package, "source")
        raw_authors = raw_package.get("authors")
        authors = (
            tuple(author for author in raw_authors if isinstance(author, str))
            if isinstance(raw_authors, list)
            else ()
        )
        declared_license = raw_package.get("license")
        if not isinstance(declared_license, str):
            license_file = raw_package.get("license_file")
            if not isinstance(license_file, str):
                raise RuntimeError(f"{name} {version} has no declared license")
            declared_license = f"LicenseRef-File:{Path(license_file).name}"

        document_ids: list[str] = []
        for path in candidate_license_files(raw_package):
            text = normalize_license(path)
            digest = hashlib.sha256(text.encode()).hexdigest()
            document_id = digest[:16]
            documents[document_id] = text
            document_users[document_id].add(f"{name} {version}")
            document_ids.append(document_id)
        records.append(
            Package(
                name=name,
                version=version,
                source=source,
                declared_license=declared_license,
                authors=authors,
                license_documents=tuple(document_ids),
            )
        )

    compiler_package, compiler_document_id, compiler_license = compiler_builtins()
    records.append(compiler_package)
    documents[compiler_document_id] = compiler_license
    document_users[compiler_document_id].add(
        f"{compiler_package.name} {compiler_package.version}"
    )
    records.sort(key=lambda package: (package.name.casefold(), package.version, package.source))
    return records, documents, document_users


def markdown_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render() -> str:
    packages, documents, document_users = dependency_packages(cargo_metadata())
    lines = [
        "# Locked Rust dependency licenses",
        "",
        "<!-- Generated by tools/update_rust_licenses.py. Do not edit manually. -->",
        "",
        "This bundle covers every non-workspace Rust package selected by the checked-in",
        "`Cargo.lock`, including target-specific, development, and build dependencies.",
        "Declared expressions and authors come from package metadata. Exact license",
        "documents come from corresponding package archives when those archives contain",
        "one; a dash marks packages whose archive relies on its metadata declaration.",
        "",
        f"Packages: {len(packages)}. Unique license documents: {len(documents)}.",
        "",
        "| Package | Version | Declared license | Authors | Documents | Source |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for package in packages:
        document_links = ", ".join(
            f"[`{document_id}`](#{document_id})"
            for document_id in package.license_documents
        ) or "—"
        authors = ", ".join(package.authors) or "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{markdown_escape(package.name)}`",
                    f"`{markdown_escape(package.version)}`",
                    f"`{markdown_escape(package.declared_license)}`",
                    markdown_escape(authors),
                    document_links,
                    f"`{markdown_escape(package.source)}`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## License documents", ""])
    for document_id in sorted(documents):
        users = ", ".join(f"`{user}`" for user in sorted(document_users[document_id]))
        lines.extend(
            [
                f'<a id="{document_id}"></a>',
                "",
                f"### `{document_id}`",
                "",
                f"Used by: {users}.",
                "",
            ]
        )
        lines.extend(f"    {line}" if line else "" for line in documents[document_id].splitlines())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = arguments()
    try:
        generated = render()
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.check:
        try:
            current = OUTPUT.read_text(encoding="utf-8")
        except OSError as error:
            print(f"error: cannot read {OUTPUT}: {error}", file=sys.stderr)
            return 1
        if current != generated:
            print(
                "error: Rust dependency license bundle is stale; run "
                "tools/update_rust_licenses.py",
                file=sys.stderr,
            )
            return 1
        return 0

    try:
        OUTPUT.write_text(generated, encoding="utf-8")
    except OSError as error:
        print(f"error: cannot write {OUTPUT}: {error}", file=sys.stderr)
        return 1
    print(f"updated {OUTPUT.relative_to(REPOSITORY_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
