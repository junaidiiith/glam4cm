#!/usr/bin/env python3
"""Package local datasets and upload them as GitHub release assets."""

import argparse
import json
import mimetypes
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


DATASETS = ("ecore_555", "modelset", "eamodelset")
DEFAULT_REPO = os.environ.get("GITHUB_REPOSITORY", "junaidiiith/glam4cm")
DEFAULT_TAG = "datasets-v1"
DEFAULT_TITLE = "GLAM4CM datasets"
API_ROOT = "https://api.github.com"
UPLOAD_ROOT = "https://uploads.github.com"


class GitHubError(RuntimeError):
    pass


def permission_hint(status_code: int, message: str) -> str:
    if status_code != 403:
        return ""
    if "Resource not accessible by personal access token" not in message:
        return ""
    return (
        "\n\nToken permission hint: this operation creates or updates GitHub releases. "
        "For a fine-grained personal access token, grant access to repository "
        "junaidiiith/glam4cm and set Repository permissions -> Contents: Read and write. "
        "For a classic token, use the repo scope for a private repo or public_repo for a public repo. "
        "If you are using a GitHub Actions GITHUB_TOKEN, set permissions: contents: write."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zip GLAM4CM datasets and upload them to a GitHub release."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo. Default: {DEFAULT_REPO}.")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag. Default: {DEFAULT_TAG}.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help=f"Release title. Default: {DEFAULT_TITLE}.")
    parser.add_argument("--datasets-dir", default="datasets", help="Directory containing dataset folders.")
    parser.add_argument("--output-dir", default=None, help="Keep generated zip files in this directory.")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token. Defaults to GITHUB_TOKEN.")
    parser.add_argument("--clobber", action="store_true", help="Replace existing release assets with the same names.")
    parser.add_argument("--dry-run", action="store_true", help="Package/check paths but do not call GitHub.")
    parser.epilog = (
        "Token permissions: fine-grained PAT needs repository access plus "
        "Repository permissions -> Contents: Read and write. Classic PAT needs "
        "repo for private repositories or public_repo for public repositories."
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def request_json(
    method: str,
    url: str,
    token: Optional[str],
    payload: Optional[Dict[str, object]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "glam4cm-dataset-release-script",
    }
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            content = response.read()
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise GitHubError(
            f"GitHub API {method} {url} failed: HTTP {exc.code}: {message}"
            f"{permission_hint(exc.code, message)}"
        ) from exc

    if not content:
        return {}
    return json.loads(content.decode("utf-8"))


def github_api(method: str, repo: str, path: str, token: Optional[str], payload=None) -> Dict[str, object]:
    return request_json(method, f"{API_ROOT}/repos/{repo}{path}", token, payload)


def get_release(repo: str, tag: str, token: Optional[str]) -> Optional[Dict[str, object]]:
    url = f"{API_ROOT}/repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe='')}"
    try:
        return request_json("GET", url, token)
    except GitHubError as exc:
        if "HTTP 404" in str(exc):
            return None
        raise


def ensure_release(repo: str, tag: str, title: str, token: str) -> Dict[str, object]:
    release = get_release(repo, tag, token)
    if release:
        print(f"Release {tag} already exists in {repo}")
        return release

    print(f"Creating release {tag} in {repo}")
    return github_api(
        "POST",
        repo,
        "/releases",
        token,
        {
            "tag_name": tag,
            "name": title,
            "body": "Dataset release for GLAM4CM.",
        },
    )


def asset_by_name(release: Dict[str, object], name: str) -> Optional[Dict[str, object]]:
    for asset in release.get("assets", []):
        if asset.get("name") == name:
            return asset
    return None


def delete_asset(repo: str, asset_id: int, token: str) -> None:
    github_api("DELETE", repo, f"/releases/assets/{asset_id}", token)


def upload_asset(repo: str, release: Dict[str, object], asset_path: Path, token: str) -> Dict[str, object]:
    upload_url = f"{UPLOAD_ROOT}/repos/{repo}/releases/{release['id']}/assets"
    query = urllib.parse.urlencode({"name": asset_path.name})
    content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "Content-Type": content_type,
        "User-Agent": "glam4cm-dataset-release-script",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = asset_path.read_bytes()
    request = urllib.request.Request(
        f"{upload_url}?{query}",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise GitHubError(
            f"GitHub asset upload failed for {asset_path.name}: HTTP {exc.code}: {message}"
            f"{permission_hint(exc.code, message)}"
        ) from exc


def zip_dataset(dataset_root: Path, dataset_name: str, asset_dir: Path) -> Path:
    source = dataset_root / dataset_name
    if not source.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {source}")

    asset_path = asset_dir / f"{dataset_name}.zip"
    if asset_path.exists():
        asset_path.unlink()

    print(f"Packaging {source} -> {asset_path}")
    with zipfile.ZipFile(asset_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(dataset_root))
    return asset_path


def build_assets(dataset_root: Path, asset_dir: Path) -> List[Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    return [zip_dataset(dataset_root, dataset, asset_dir) for dataset in DATASETS]


def main() -> int:
    args = parse_args()
    root = repo_root()
    dataset_root = Path(args.datasets_dir)
    if not dataset_root.is_absolute():
        dataset_root = root / dataset_root

    temp_dir = None
    if args.output_dir:
        asset_dir = Path(args.output_dir)
        if not asset_dir.is_absolute():
            asset_dir = root / asset_dir
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="glam4cm-datasets-"))
        asset_dir = temp_dir

    try:
        assets = build_assets(dataset_root, asset_dir)
        if args.dry_run:
            print(f"Would ensure release {args.tag} in {args.repo}")
            for asset in assets:
                print(f"Would upload {asset}")
            return 0

        if not args.token:
            raise SystemExit("Missing GitHub token. Set GITHUB_TOKEN or pass --token.")

        release = ensure_release(args.repo, args.tag, args.title, args.token)
        release = get_release(args.repo, args.tag, args.token) or release
        for asset_path in assets:
            existing = asset_by_name(release, asset_path.name)
            if existing:
                if not args.clobber:
                    raise SystemExit(
                        f"Asset already exists: {asset_path.name}. Pass --clobber to replace it."
                    )
                print(f"Deleting existing asset {asset_path.name}")
                delete_asset(args.repo, int(existing["id"]), args.token)

            print(f"Uploading {asset_path.name}")
            upload_asset(args.repo, release, asset_path, args.token)

        print("Upload complete.")
        return 0
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
