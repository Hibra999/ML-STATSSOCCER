from pathlib import PurePosixPath

from fastapi.staticfiles import StaticFiles


class PublicStorageAssets(StaticFiles):
    """Serve only public asset folders from storage."""

    _allowed_roots = {"graphics"}

    def lookup_path(self, path: str):
        public_path = str(path or "").lstrip("/")
        parts = PurePosixPath(public_path).parts
        if not parts or parts[0] not in self._allowed_roots:
            return "", None
        return super().lookup_path(public_path)
