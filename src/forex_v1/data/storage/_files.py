"""Atomic immutable file creation shared by local storage backends."""

import os
import tempfile
from pathlib import Path


class ImmutableFileError(RuntimeError):
    """Raised when a file cannot be safely created without overwriting data."""


def write_immutable_file(path: Path, content: bytes) -> None:
    """Atomically create a file, accepting an existing identical file only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"Refusing to overwrite different immutable file: {path}")
    except OSError as error:
        if isinstance(error, FileExistsError):
            raise
        raise ImmutableFileError(f"Could not atomically write immutable file {path}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
