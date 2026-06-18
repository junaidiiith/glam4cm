#!/usr/bin/env python3
"""Download GLAM4CM datasets from GitHub release assets."""

import argparse
import json
import os
import shutil
import sys
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
API_ROOT = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract GLAM4CM datasets from GitHub release assets."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"GitHub repo. Default: {DEFAULT_REPO}.")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"Release tag. Default: {DEFAULT_TAG}.")
    parser.add_argument("--dest-dir", default="datasets", help="Extraction destination. Default: datasets.")
    parser.add_argument("--download-dir", default=None, help="Keep downloaded zip files in this directory.")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"), help="Optional GitHub token. Defaults to GITHUB_TOKEN.")
    parser.add_argument("--clobber", action="store_true", help="Replace existing dataset directories.")
    parser.add_argument("--keep-zips", action="store_true", help="Keep zip files when using a temp download dir.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def request(
    method: str,
    url: str,
    token: Optional[str],
    accept: str = "application/vnd.github+json",
) -> bytes:
    headers = {
        "Accept": accept,
        "User-Agent": "glam4cm-dataset-release-script",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise GitHubError(f"GitHub request failed: HTTP {exc.code}: {message}") from exc


def request_json(url: str, token: Optional[str]) -> Dict[str, object]:
    return json.loads(request("GET", url, token).decode("utf-8"))


def release_by_tag(repo: str, tag: str, token: Optional[str]) -> Dict[str, object]:
    quoted_tag = urllib.parse.quote(tag, safe="")
    return request_json(f"{API_ROOT}/repos/{repo}/releases/tags/{quoted_tag}", token)


def asset_map(release: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        str(asset["name"]): asset
        for asset in release.get("assets", [])
    }


def download_asset(asset: Dict[str, object], destination: Path, token: Optional[str]) -> None:
    if token:
        data = request("GET", str(asset["url"]), token, accept="application/octet-stream")
    else:
        data = request("GET", str(asset["browser_download_url"]), token, accept="application/octet-stream")
    destination.write_bytes(data)


def extract_dataset(zip_path: Path, dest_dir: Path, dataset_name: str, clobber: bool) -> None:
    target = dest_dir / dataset_name
    if target.exists():
        if not clobber:
            raise SystemExit(f"Destination already exists: {target}. Pass --clobber to replace it.")
        shutil.rmtree(target)

    print(f"Extracting {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest_dir)


def main() -> int:
    args = parse_args()
    root = repo_root()

    dest_dir = Path(args.dest_dir)
    if not dest_dir.is_absolute():
        dest_dir = root / dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = None
    if args.download_dir:
        download_dir = Path(args.download_dir)
        if not download_dir.is_absolute():
            download_dir = root / download_dir
        download_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="glam4cm-dataset-downloads-"))
        download_dir = temp_dir

    try:
        release = release_by_tag(args.repo, args.tag, args.token)
        assets = asset_map(release)

        for dataset in DATASETS:
            asset_name = f"{dataset}.zip"
            if asset_name not in assets:
                raise SystemExit(f"Release {args.tag} does not contain asset: {asset_name}")

            zip_path = download_dir / asset_name
            print(f"Downloading {asset_name} from {args.repo} release {args.tag}")
            download_asset(assets[asset_name], zip_path, args.token)
            extract_dataset(zip_path, dest_dir, dataset, args.clobber)

        print("Download complete.")
        if args.download_dir or args.keep_zips:
            print(f"Zip files kept in {download_dir}")
        return 0
    finally:
        if temp_dir is not None and not args.keep_zips:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
