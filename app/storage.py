"""Object storage. `local` backend for dev/tests, `s3` for AWS S3 (or MinIO via S3_ENDPOINT_URL)."""

import hashlib
import hmac
import math
import os
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import quote

from app.config import Settings, get_settings


class StorageError(Exception):
    pass


class Storage(ABC):
    supports_multipart = False

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    @abstractmethod
    def read_head(self, key: str, n: int = 4096) -> bytes: ...

    @abstractmethod
    def download_to(self, key: str, dest: Path) -> None: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def presign_upload(self, key: str, declared_size: int, max_bytes: int, content_type: str) -> dict:
        """Returns {method,url,fields,headers,expires_in,multipart}. Server never receives the bytes."""

    def complete_multipart(self, key: str, upload_id: str, parts: list[dict]) -> None:
        raise StorageError("multipart not supported by this backend")


class LocalStorage(Storage):
    def __init__(self, root: str, settings: Settings):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings = settings

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise StorageError("invalid key")
        return p

    def put_bytes(self, key, data):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def get_bytes(self, key):
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as exc:
            raise StorageError(f"missing object: {key}") from exc

    def exists(self, key):
        return self._path(key).is_file()

    def size(self, key):
        try:
            return self._path(key).stat().st_size
        except FileNotFoundError as exc:
            raise StorageError(f"missing object: {key}") from exc

    def read_head(self, key, n=4096):
        with self._path(key).open("rb") as f:
            return f.read(n)

    def download_to(self, key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(self._path(key), dest)
        except FileNotFoundError as exc:
            raise StorageError(f"missing object: {key}") from exc

    def delete_prefix(self, prefix):
        p = self._path(prefix)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)

    def sign(self, key: str, exp: int, max_bytes: int) -> str:
        msg = f"{key}|{exp}|{max_bytes}".encode()
        return hmac.new(self.settings.upload_signing_secret.encode(), msg, hashlib.sha256).hexdigest()

    def verify(self, key: str, exp: int, max_bytes: int, sig: str) -> bool:
        return time.time() < exp and hmac.compare_digest(self.sign(key, exp, max_bytes), sig)

    def presign_upload(self, key, declared_size, max_bytes, content_type):
        exp = int(time.time()) + self.settings.presign_ttl_s
        sig = self.sign(key, exp, max_bytes)
        url = (
            f"{self.settings.public_base_url}/api/v1/_local/upload"
            f"?key={quote(key, safe='')}&exp={exp}&max={max_bytes}&sig={sig}"
        )
        return {
            "method": "PUT",
            "url": url,
            "fields": {},
            "headers": {"Content-Type": content_type},
            "expires_in": self.settings.presign_ttl_s,
            "multipart": None,
        }


class S3Storage(Storage):
    supports_multipart = True

    def __init__(self, settings: Settings):
        import boto3
        from botocore.config import Config

        self.settings = settings
        self.bucket = settings.s3_bucket
        self.client = boto3.client(
            "s3",
            region_name=settings.s3_region,
            endpoint_url=settings.s3_endpoint_url,
            config=Config(
                signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"},
                # boto3 >= 1.36 adds CRC checksums by default, which Cloudflare R2 and other S3-compatibles reject
                request_checksum_calculation="when_required", response_checksum_validation="when_required",
            ),
        )

    def put_bytes(self, key, data):
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get_bytes(self, key):
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey as exc:
            raise StorageError(f"missing object: {key}") from exc

    def exists(self, key):
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.client.exceptions.ClientError:
            return False

    def size(self, key):
        try:
            return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])
        except self.client.exceptions.ClientError as exc:
            raise StorageError(f"missing object: {key}") from exc

    def read_head(self, key, n=4096):
        resp = self.client.get_object(Bucket=self.bucket, Key=key, Range=f"bytes=0-{n - 1}")
        return resp["Body"].read()

    def download_to(self, key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))

    def delete_prefix(self, prefix):
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})

    def delete(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def presign_upload(self, key, declared_size, max_bytes, content_type):
        s = self.settings
        if declared_size > s.multipart_threshold_bytes:
            part_size = s.multipart_part_size
            n_parts = math.ceil(declared_size / part_size)
            if n_parts > 10_000:
                part_size = math.ceil(declared_size / 10_000)
                n_parts = math.ceil(declared_size / part_size)
            upload_id = self.client.create_multipart_upload(Bucket=self.bucket, Key=key, ContentType=content_type)["UploadId"]
            parts = [
                {
                    "part_number": n,
                    "url": self.client.generate_presigned_url(
                        "upload_part",
                        Params={"Bucket": self.bucket, "Key": key, "UploadId": upload_id, "PartNumber": n},
                        ExpiresIn=max(s.presign_ttl_s, 3600),
                    ),
                }
                for n in range(1, n_parts + 1)
            ]
            return {
                "method": "PUT",
                "url": None,
                "fields": {},
                "headers": {},
                "expires_in": max(s.presign_ttl_s, 3600),
                "multipart": {"upload_id": upload_id, "part_size": part_size, "parts": parts},
            }
        if s.s3_presign_mode == "put":
            # R2 has no POST policies. Signing Content-Length pins the upload to the size the student declared
            # (any other body length fails signature verification); /uploads/complete re-checks size and magic bytes.
            url = self.client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type, "ContentLength": declared_size},
                ExpiresIn=s.presign_ttl_s,
            )
            return {"method": "PUT", "url": url, "fields": {}, "headers": {"Content-Type": content_type}, "expires_in": s.presign_ttl_s, "multipart": None}
        # Presigned POST lets S3 itself enforce the size ceiling (a presigned PUT cannot).
        post = self.client.generate_presigned_post(
            Bucket=self.bucket,
            Key=key,
            Fields={"Content-Type": content_type},
            Conditions=[["content-length-range", 1, max_bytes], {"Content-Type": content_type}],
            ExpiresIn=s.presign_ttl_s,
        )
        return {
            "method": "POST",
            "url": post["url"],
            "fields": post["fields"],
            "headers": {},
            "expires_in": s.presign_ttl_s,
            "multipart": None,
        }

    def complete_multipart(self, key, upload_id, parts):
        ordered = sorted(({"PartNumber": int(p["part_number"]), "ETag": p["etag"]} for p in parts), key=lambda p: p["PartNumber"])
        self.client.complete_multipart_upload(
            Bucket=self.bucket, Key=key, UploadId=upload_id, MultipartUpload={"Parts": ordered}
        )

    def configure_bucket(self) -> dict[str, str]:
        """Abort dangling multipart uploads (silent cost leak), expire old objects, allow browser uploads.
        Each step reports independently: S3-compatibles implement different subsets of the bucket API."""
        from botocore.exceptions import ClientError

        steps = {
            "lifecycle": lambda: self.client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket,
                LifecycleConfiguration={
                    "Rules": [
                        {"ID": "abort-incomplete-multipart", "Status": "Enabled", "Filter": {"Prefix": ""}, "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 2}},
                        {"ID": "expire-uploads", "Status": "Enabled", "Filter": {"Prefix": "uploads/"}, "Expiration": {"Days": 90}},
                        {"ID": "expire-job-artifacts", "Status": "Enabled", "Filter": {"Prefix": "jobs/"}, "Expiration": {"Days": 180}},
                    ]
                },
            ),
            "cors": lambda: self.client.put_bucket_cors(
                Bucket=self.bucket,
                CORSConfiguration={
                    "CORSRules": [
                        {
                            "AllowedMethods": ["PUT", "POST"],
                            "AllowedOrigins": [o.strip() for o in (self.settings.cors_origins or self.settings.public_base_url).split(",")],
                            "AllowedHeaders": ["*"],
                            "ExposeHeaders": ["ETag"],
                            "MaxAgeSeconds": 3000,
                        }
                    ]
                },
            ),
        }
        results = {}
        for name, call in steps.items():
            try:
                call()
                results[name] = "ok"
            except ClientError as exc:
                results[name] = f"failed: {exc.response.get('Error', {}).get('Code', 'error')} - set it in the provider's dashboard"
        return results


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        s = get_settings()
        _storage = S3Storage(s) if s.storage_backend == "s3" else LocalStorage(s.local_storage_dir, s)
    return _storage


def set_storage(storage: Storage | None) -> None:
    global _storage
    _storage = storage
