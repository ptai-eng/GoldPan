"""Repository file walker respecting .gitignore, size limits, binary detection, and path filters."""

import os
from collections.abc import Generator
from pathlib import Path

import pathspec

DEFAULT_IGNORE_DIRS: set[str] = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".next",
    "dist",
    "build",
    "target",
    ".chroma",
    "chroma_data",
}


def is_binary_file(file_path: Path, chunk_size: int = 8192) -> bool:
    """Check if file is binary by inspecting initial bytes for NUL characters."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(chunk_size)
            if b"\x00" in chunk:
                return True
        return False
    except Exception:
        return True


class RepoWalker:
    """Recursively walks a repository directory respecting .gitignore and file filters."""

    def __init__(
        self,
        root_dir: str | Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_kb: int = 1024,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.max_file_bytes = max_file_kb * 1024

        self.root_spec = self._load_root_gitignore()
        self.user_include_spec = (
            pathspec.PathSpec.from_lines("gitignore", self.include_patterns)
            if self.include_patterns
            else None
        )
        self.user_exclude_spec = (
            pathspec.PathSpec.from_lines("gitignore", self.exclude_patterns)
            if self.exclude_patterns
            else None
        )

    def _load_root_gitignore(self) -> pathspec.PathSpec | None:
        gi_path = self.root_dir / ".gitignore"
        if gi_path.is_file():
            try:
                lines = gi_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
                return pathspec.PathSpec.from_lines("gitignore", lines)
            except Exception:
                return None
        return None

    def walk(self) -> Generator[tuple[Path, str], None, tuple[int, list[str]]]:
        """
        Yields (absolute_path, relative_posix_path).
        Returns a tuple of (files_skipped_count, residual_unknowns).
        """
        files_skipped = 0
        residual_unknowns: list[str] = []
        visited_inodes: set[tuple[int, int]] = set()

        # Track directory gitignores for nested .gitignore support
        dir_specs: dict[Path, pathspec.PathSpec | None] = {
            self.root_dir: self.root_spec
        }

        for dirpath, dirnames, filenames in os.walk(
            str(self.root_dir), followlinks=False
        ):
            current_dir = Path(dirpath).resolve()

            # Prevent outside-root symlink escapes
            try:
                current_dir.relative_to(self.root_dir)
            except ValueError:
                dirnames.clear()
                continue

            # Symlink loop protection via inode check (on Unix / NTFS)
            try:
                stat = current_dir.stat()
                inode_key = (stat.st_ino, stat.st_dev)
                if stat.st_ino and inode_key in visited_inodes:
                    dirnames.clear()
                    continue
                if stat.st_ino:
                    visited_inodes.add(inode_key)
            except OSError:
                pass

            # Prune default ignored directories
            dirnames[:] = [
                d
                for d in dirnames
                if d not in DEFAULT_IGNORE_DIRS
                and not d.startswith(".git")
                and (
                    not self.root_spec
                    or not self.root_spec.match_file(
                        str((current_dir / d).relative_to(self.root_dir)).replace(
                            "\\", "/"
                        )
                        + "/"
                    )
                )
            ]

            # Load nested .gitignore if present
            nested_gi = current_dir / ".gitignore"
            if nested_gi != (self.root_dir / ".gitignore") and nested_gi.is_file():
                try:
                    lines = nested_gi.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                    dir_specs[current_dir] = pathspec.PathSpec.from_lines(
                        "gitignore", lines
                    )
                except Exception:
                    dir_specs[current_dir] = None

            for fname in filenames:
                full_path = current_dir / fname
                try:
                    rel_posix = str(full_path.relative_to(self.root_dir)).replace(
                        "\\", "/"
                    )
                except ValueError:
                    continue

                # 1. Check root .gitignore
                if self.root_spec and self.root_spec.match_file(rel_posix):
                    files_skipped += 1
                    continue

                # 2. Check nested .gitignore
                if dir_specs.get(current_dir):
                    nested_rel = str(full_path.relative_to(current_dir)).replace(
                        "\\", "/"
                    )
                    if dir_specs[current_dir].match_file(nested_rel):
                        files_skipped += 1
                        continue

                # 3. Check user exclude patterns
                if self.user_exclude_spec and self.user_exclude_spec.match_file(
                    rel_posix
                ):
                    files_skipped += 1
                    continue

                # 4. Check user include patterns
                if self.user_include_spec and not self.user_include_spec.match_file(
                    rel_posix
                ):
                    files_skipped += 1
                    continue

                # 5. Check file size cap
                try:
                    size = full_path.stat().st_size
                    if size > self.max_file_bytes:
                        files_skipped += 1
                        residual_unknowns.append(
                            f"Skipped oversize file ({size // 1024} KB > {self.max_file_bytes // 1024} KB): {rel_posix}"
                        )
                        continue
                except OSError:
                    files_skipped += 1
                    residual_unknowns.append(f"Unreadable file metadata: {rel_posix}")
                    continue

                # 6. Check binary
                if is_binary_file(full_path):
                    files_skipped += 1
                    residual_unknowns.append(f"Skipped binary file: {rel_posix}")
                    continue

                yield full_path, rel_posix

        return files_skipped, residual_unknowns
