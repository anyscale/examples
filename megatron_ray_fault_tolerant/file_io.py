"""
File I/O utilities for handling both local filesystem and cloud storage (S3/GCS).

This module provides a unified interface for file operations that works with:
- Local filesystem paths
- S3 paths (s3://bucket/path)
- Google Cloud Storage paths (gs://bucket/path or gcs://bucket/path)

Uses fsspec for cloud storage abstraction.
"""

import os
import tempfile
from contextlib import contextmanager
import fsspec
from loguru import logger
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from timer import Timer
# Optional AWS deps (present when s3fs is installed)
try:
    import botocore.session as _botocore_session
    from botocore.exceptions import ClientError

    _HAS_BOTOCORE = True
except Exception:
    _HAS_BOTOCORE = False

    class ClientError(Exception):  # fallback type
        pass


_S3_FS = None  # type: ignore


def get_s3_fs():
    """Return a cached S3 filesystem instance, creating it once."""
    global _S3_FS
    if _S3_FS is None:
        _S3_FS = fsspec.filesystem("s3")
    return _S3_FS


def s3_expiry_time():
    """Return botocore credential expiry (datetime in UTC) or None."""
    if not _HAS_BOTOCORE:
        return None
    try:
        sess = _botocore_session.get_session()
        creds = sess.get_credentials()
        if not creds:
            return None
        return getattr(creds, "expiry_time", None) or getattr(
            creds, "_expiry_time", None
        )
    except Exception:
        return None


def s3_refresh_if_expiring(fs) -> None:
    """
    Simple refresh:
    - If expiry exists and is within 300s (or past), refresh with fs.connect(refresh=True).
    - Otherwise, do nothing.
    """
    exp = s3_expiry_time()
    if not exp:
        return
    now = datetime.now(timezone.utc)
    if now >= exp - timedelta(seconds=300):
        try:
            fs.connect(refresh=True)  # rebuild session
        except Exception:
            pass


def call_with_s3_retry(fs, fn, *args, **kwargs):
    """
    Wrapper for calling an S3 method. If it fails with ExpiredToken, force refresh once and retry.
    """
    try:
        return fn(*args, **kwargs)
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code")
        if code in {
            "ExpiredToken",
            "ExpiredTokenException",
            "RequestExpired",
        } and hasattr(fs, "connect"):
            try:
                fs.connect(refresh=True)
            except Exception:
                pass
            return fn(*args, **kwargs)
        raise


def is_cloud_path(path: str) -> bool:
    """Check if the given path is a cloud storage path."""
    return path.startswith(("s3://", "gs://", "gcs://"))


def _get_filesystem(path: str):
    """Get the appropriate filesystem for the given path."""
    if not is_cloud_path(path):
        return fsspec.filesystem("file")

    proto = path.split("://", 1)[0]
    if proto == "s3":
        fs = get_s3_fs()
        s3_refresh_if_expiring(fs)
        return fs
    return fsspec.filesystem(proto)


def open_file(path: str, mode: str = "rb"):
    """Open a file using fsspec, works with both local and cloud paths."""
    if not is_cloud_path(path):
        return fsspec.open(path, mode)

    fs = _get_filesystem(path)
    norm = fs._strip_protocol(path)
    try:
        return fs.open(norm, mode)
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code")
        if code in {
            "ExpiredToken",
            "ExpiredTokenException",
            "RequestExpired",
        } and hasattr(fs, "connect"):
            try:
                fs.connect(refresh=True)
            except Exception:
                pass
            return fs.open(norm, mode)
        raise


def makedirs(path: str, exist_ok: bool = True) -> None:
    """Create directories. Only applies to local filesystem paths."""
    if not is_cloud_path(path):
        os.makedirs(path, exist_ok=exist_ok)


def exists(path: str) -> bool:
    """Check if a file or directory exists."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.exists, path)
    return fs.exists(path)


def isdir(path: str) -> bool:
    """Check if path is a directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.isdir, path)
    return fs.isdir(path)


def list_dir(path: str) -> list[str]:
    """List contents of a directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.ls, path, detail=False)
    return fs.ls(path, detail=False)


def remove(path: str) -> None:
    """Remove a file or directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        if call_with_s3_retry(fs, fs.isdir, path):
            call_with_s3_retry(fs, fs.rm, path, recursive=True)
        else:
            call_with_s3_retry(fs, fs.rm, path)
        return
    if fs.isdir(path):
        fs.rm(path, recursive=True)
    else:
        fs.rm(path)


def upload_directory(local_path: str, cloud_path: str) -> None:
    """Upload a local directory to cloud storage.

    Uploads the contents of local_path to cloud_path, not the directory itself.
    This ensures consistent behavior across all ranks by explicitly uploading each file.
    """
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Destination must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)

    # Normalize paths: ensure cloud_path ends with / to indicate directory
    cloud_path_normalized = cloud_path.rstrip("/") + "/"

    # Walk the local directory and upload each file explicitly
    # This ensures we upload contents, not the directory as a subdirectory
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Get relative path from local_path to maintain directory structure
            rel_path = os.path.relpath(local_file_path, local_path)
            # Construct remote path: cloud_path/rel_path
            remote_file_path = cloud_path_normalized + rel_path

            if cloud_path.startswith("s3://"):
                # For S3, strip protocol for fsspec operations
                remote_file_path_stripped = fs._strip_protocol(remote_file_path)
                # Ensure parent directories exist in S3 (fsspec handles this automatically)
                call_with_s3_retry(
                    fs, fs.put, local_file_path, remote_file_path_stripped
                )
            else:
                fs.put(local_file_path, remote_file_path)

    logger.info(f"Uploaded contents of {local_path} to {cloud_path}")


def download_directory(cloud_path: str, local_path: str, prefixes: List[str] = None) -> None:
    """Download a cloud directory to local storage."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Source must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    cloud_path_normalized = cloud_path.rstrip("/") + "/"
    os.makedirs(local_path, exist_ok=True)

    # List all files and download each one individually to download contents, not the folder
    if cloud_path.startswith("s3://"):
        remote_path_stripped = fs._strip_protocol(cloud_path_normalized)
        all_files = call_with_s3_retry(fs, fs.find, remote_path_stripped, detail=False)
        for remote_file in all_files:
            if remote_file.endswith("/"):
                continue
            rel_path = remote_file[len(remote_path_stripped) :].lstrip("/")
            if prefixes and not any(rel_path.startswith(prefix) for prefix in prefixes):
                continue
            local_file_path = os.path.join(local_path, rel_path)
            parent_dir = os.path.dirname(local_file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            call_with_s3_retry(fs, fs.get, remote_file, local_file_path)
    else:
        all_files = fs.find(cloud_path_normalized, detail=False)
        for remote_file in all_files:
            if remote_file.endswith("/"):
                continue
            rel_path = remote_file[len(cloud_path_normalized) :].lstrip("/")
            if prefixes and not any(rel_path.startswith(prefix) for prefix in prefixes):
                continue
            local_file_path = os.path.join(local_path, rel_path)
            parent_dir = os.path.dirname(local_file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            fs.get(remote_file, local_file_path)

    logger.info(f"Downloaded {cloud_path} to {local_path}")


@contextmanager
def local_work_dir(output_path: str):
    """
    Context manager that provides a local working directory.

    For local paths, returns the path directly.
    For cloud paths, creates a temporary directory and uploads content at the end.

    Args:
        output_path: The final destination path (local or cloud)

    Yields:
        str: Local directory path to work with

    Example:
        with local_work_dir("s3://bucket/model") as work_dir:
            # Save files to work_dir
            model.save_pretrained(work_dir)
            # Files are automatically uploaded to s3://bucket/model at context exit
    """
    if is_cloud_path(output_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                yield temp_dir
            finally:
                # Upload everything from temp_dir to cloud path
                with Timer("Uploading directory contents to cloud path..."):
                    upload_directory(temp_dir, output_path)
                    logger.info(f"Uploaded directory contents to {output_path}")
    else:
        # For local paths, ensure directory exists and use it directly
        makedirs(output_path, exist_ok=True)
        yield output_path


@contextmanager
def local_read_dir(input_path: str, local_path: Optional[str] = None, prefixes: List[str] = None):
    """
    Context manager that provides a local directory with content from input_path.

    For local paths, returns the path directly.
    For cloud paths, downloads content to a temporary directory.

    Args:
        input_path: The source path (local or cloud)
        local_path: The local path to download the directory to. If None, use a temporary directory.
        prefixes: If using cloud path, only download files that start with any of these prefixes. If None, download all files.

    Yields:
        str: Local directory path containing the content

    Example:
        with local_read_dir("s3://bucket/model") as read_dir:
            # Load files from read_dir
            model = AutoModel.from_pretrained(read_dir)
    """
    if is_cloud_path(input_path):
        if local_path is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download everything from cloud path to temp_dir
                download_directory(input_path, temp_dir, prefixes)
                logger.info(f"Downloaded directory contents from {input_path}")
                yield temp_dir
        else:
            # Download everything from cloud path to local_path
            with Timer("Downloading directory contents to cloud path..."):
                download_directory(input_path, local_path, prefixes)
                logger.info(f"Downloaded directory contents from {input_path}")
                yield local_path
    else:
        # For local paths, use directly (but check it exists)
        if not exists(input_path):
            raise FileNotFoundError(f"Path does not exist: {input_path}")
        yield input_path
