import os
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


_MODELSCOPE_BASE = "https://www.modelscope.cn"


def _download_url(url, dst_path, timeout=60):
    """Download *url* into *dst_path* atomically."""
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    tmp_path = dst_path + ".tmp"
    with urlopen(url, timeout=timeout) as response, open(tmp_path, "wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(tmp_path, dst_path)
    return dst_path


def _modelscope_urls(repo_type, repo_id, filename):
    base = "models" if repo_type == "model" else "datasets"
    return [
        f"{_MODELSCOPE_BASE}/{base}/{repo_id}/resolve/master/{filename}",
        f"{_MODELSCOPE_BASE}/{base}/{repo_id}/resolve/main/{filename}",
    ]


def download_from_modelscope(repo_id, filename, local_dir, repo_type="model"):
    """Download a file from ModelScope into *local_dir*."""
    local_path = os.path.join(local_dir, os.path.basename(filename))
    errors = []
    for url in _modelscope_urls(repo_type, repo_id, filename):
        try:
            print(f"Downloading {os.path.basename(filename)} from ModelScope...", file=sys.stderr)
            return _download_url(url, local_path)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("ModelScope download failed:\n" + "\n".join(errors))


def resolve_local_then_download(
    path_arg,
    *,
    local_dir,
    default_filename,
    hf_repo_id,
    hf_repo_type="model",
    ms_repo_id=None,
    ms_repo_type="model",
):
    """Resolve a local artifact path, or download from ModelScope/HF."""
    os.makedirs(local_dir, exist_ok=True)

    if path_arg:
        expanded = os.path.expanduser(path_arg)
        if os.path.isfile(expanded):
            print(f"Using local artifact: {expanded}", file=sys.stderr)
            return expanded
        filename = os.path.basename(path_arg)
    else:
        filename = default_filename
        default_local = os.path.join(local_dir, filename)
        if os.path.isfile(default_local):
            print(f"Using cached artifact: {default_local}", file=sys.stderr)
            return default_local

    if ms_repo_id:
        try:
            return download_from_modelscope(
                repo_id=ms_repo_id,
                filename=filename,
                local_dir=local_dir,
                repo_type=ms_repo_type,
            )
        except Exception as exc:
            print(f"ModelScope download failed, falling back to Hugging Face: {exc}", file=sys.stderr)

    from huggingface_hub import hf_hub_download

    print(f"Downloading {filename} from {hf_repo_id}...", file=sys.stderr)
    return hf_hub_download(
        repo_id=hf_repo_id,
        filename=filename,
        repo_type=hf_repo_type,
        local_dir=local_dir,
    )
