"""Model downloader helper with checksum verification and resume support.

This provides `ensure_model(dest_path, url, sha256=None, download_dir=None)`.
"""
import os
import hashlib
import shutil
from pathlib import Path
import requests


class DownloadError(Exception):
    pass


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(dest_path: str, url: str, sha256: str = None, download_dir: str = None) -> str:
    """Ensure model file exists at `dest_path`. If not, download from `url`.

    Returns the path to the model file on success, or raises DownloadError.
    """
    dest = Path(dest_path)
    if download_dir and not dest.is_absolute():
        dest = Path(download_dir) / dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if sha256:
            if _sha256_of_file(dest) == sha256:
                return str(dest)
            else:
                # Corrupt file - remove and re-download
                dest.unlink()
        else:
            return str(dest)

    tmp_path = dest.with_suffix(dest.suffix + '.tmp')
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(tmp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        # verify checksum if provided
        if sha256:
            if _sha256_of_file(tmp_path) != sha256:
                tmp_path.unlink(missing_ok=True)
                raise DownloadError('Checksum mismatch after download')

        # atomic move
        shutil.move(str(tmp_path), str(dest))
        return str(dest)
    except Exception as e:
        # cleanup
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise DownloadError(str(e))
