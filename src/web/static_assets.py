from pathlib import PurePosixPath

from fastapi.staticfiles import StaticFiles


class PublicStorageAssets(StaticFiles):
    """Serve only public asset folders from storage."""

    _allowed_roots = {"graphics"}

    def lookup_path(self, path: str):
        parts = PurePosixPath(path).parts
        if not parts or parts[0] not in self._allowed_roots:
            return "", None
        return super().lookup_path(path)
