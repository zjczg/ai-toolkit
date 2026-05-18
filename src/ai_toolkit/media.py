"""Media utilities — upload files via SSH and build public URLs.

Independent of any AI provider and can be freely composed with API calls.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import quote

from ai_toolkit.config import get_settings


class MediaError(RuntimeError):
    """Raised when a media operation fails."""


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

    remote_target = f"{ssh_target}:{remote_dir.rstrip('/')}/"
    _run_command(["scp", *ssh_opts, str(source), remote_target])

    return f"{remote_dir.rstrip('/')}/{quote(source.name)}"


def upload_public_url(local_path: str) -> str:
    """Upload a local file and return its public HTTP(S) URL.

    Requires UPLOAD_PUBLIC_BASE_URL in addition to the SSH upload settings.
    """
    settings = get_settings()
    if not settings.upload_public_base_url:
        raise MediaError("UPLOAD_PUBLIC_BASE_URL is not configured")

    remote_path = upload_via_ssh(local_path)
    public_name = Path(remote_path).name
    return f"{settings.upload_public_base_url.rstrip('/')}/{public_name}"


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "command failed"
        raise MediaError(detail) from exc
