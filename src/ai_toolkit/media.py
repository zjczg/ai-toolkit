"""Media utilities — upload files via SSH and build public URLs.

Independent of any AI provider and can be freely composed with API calls.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen
from urllib.parse import quote

from ai_toolkit.config import get_settings


class MediaError(RuntimeError):
    """Raised when a media operation fails."""


_PUBLIC_URL_CACHE: dict[tuple[str, int, int], str] = {}


def upload_via_ssh(local_path: str) -> str:
    """Upload a local file to the configured server via SCP.

    Reads SSH config from environment variables:
        UPLOAD_SSH_TARGET     e.g. root@myhost
        UPLOAD_IDENTITY_FILE  e.g. ~/.ssh/id_ed25519
        UPLOAD_REMOTE_DIR     e.g. /var/www/images

    Returns the remote path on the server.
    """
    settings = get_settings()

    ssh_target = settings.upload_ssh_target
    if not ssh_target:
        raise MediaError("UPLOAD_SSH_TARGET is not configured")
    remote_dir = settings.upload_remote_dir
    if not remote_dir:
        raise MediaError("UPLOAD_REMOTE_DIR is not configured")

    source = Path(local_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"local file does not exist: {source}")

    identity_file = settings.upload_identity_file
    ssh_opts = ["-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes"]
    if identity_file:
        identity_path = Path(identity_file).expanduser()
        if identity_path.exists():
            ssh_opts.extend(["-i", str(identity_path)])

    _run_command(["ssh", *ssh_opts, ssh_target, f"mkdir -p {remote_dir}"])

    remote_name = unique_remote_name(source)
    remote_path = f"{remote_dir.rstrip('/')}/{remote_name}"
    remote_target = f"{ssh_target}:{remote_path}"
    _run_command(["scp", *ssh_opts, str(source), remote_target])

    return f"{remote_dir.rstrip('/')}/{quote(remote_name)}"


def upload_public_url(local_path: str) -> str:
    """Upload a local file and return its public HTTP(S) URL.

    Requires UPLOAD_PUBLIC_BASE_URL in addition to the SSH upload settings.
    """
    settings = get_settings()
    if not settings.upload_public_base_url:
        raise MediaError("UPLOAD_PUBLIC_BASE_URL is not configured")

    source = Path(local_path).expanduser().resolve()
    stat = source.stat()
    cache_key = (str(source), int(stat.st_mtime), int(stat.st_size))
    cached = _PUBLIC_URL_CACHE.get(cache_key)
    if cached:
        verify_public_url(cached)
        return cached

    remote_path = upload_via_ssh(str(source))
    public_name = Path(remote_path).name
    public_url = f"{settings.upload_public_base_url.rstrip('/')}/{public_name}"
    verify_public_url(public_url)
    _PUBLIC_URL_CACHE[cache_key] = public_url
    return public_url


def verify_public_url(url: str) -> None:
    """Ensure an uploaded media URL is publicly readable before provider use."""
    request = Request(url, method="HEAD")
    try:
        with urlopen(request, timeout=10) as response:
            status = getattr(response, "status", 200)
            if 200 <= status < 400:
                return
            raise MediaError(f"uploaded media URL is not readable: {url} returned HTTP {status}")
    except HTTPError as exc:
        raise MediaError(f"uploaded media URL is not readable: {url} returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise MediaError(f"uploaded media URL is not readable: {url}: {exc.reason}") from exc


def unique_remote_name(source: Path) -> str:
    digest = file_digest(source)[:16]
    suffix = source.suffix.lower()
    stem = source.stem or "media"
    return f"{stem}-{digest}{suffix}"


def file_digest(source: Path) -> str:
    hasher = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "command failed"
        raise MediaError(detail) from exc
