"""File storage for uploaded images and thumbnails."""

from __future__ import annotations

import hashlib
import io
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageOps

from ..config import settings

HEIF_EXTS = {".heic", ".heif"}  # iPhone photos, decoded with pillow-heif
RAW_EXTS = {".dng"}  # camera RAW, developed with rawpy (LibRaw)
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"} | HEIF_EXTS | RAW_EXTS
# Formats browsers cannot display -> converted on upload (lossless ones to PNG, photos to JPEG)
CONVERT_TO_PNG = {".bmp", ".tif", ".tiff"}
CONVERT_TO_JPG = HEIF_EXTS | RAW_EXTS

PILImage.MAX_IMAGE_PIXELS = 300_000_000  # allow large aerial/industrial images


def project_dir(project_id: int) -> Path:
    return settings.projects_dir / str(project_id)


def images_dir(project_id: int) -> Path:
    return project_dir(project_id) / "images"


def thumbs_dir(project_id: int) -> Path:
    return project_dir(project_id) / "thumbs"


def image_path(project_id: int, stored_name: str) -> Path:
    return images_dir(project_id) / stored_name


def thumb_path(project_id: int, stored_name: str) -> Path:
    return thumbs_dir(project_id) / (Path(stored_name).stem + ".jpg")


@dataclass
class StoredImage:
    stored_name: str
    width: int
    height: int
    sha1: str


class InvalidImage(ValueError):
    pass


def store_image(project_id: int, filename: str, data: bytes) -> StoredImage:
    """Validate, normalise and save an uploaded image.

    * EXIF orientation is baked into the pixels so the browser, the database
      and every training framework agree on width/height.
    * Formats browsers cannot render are converted to PNG.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise InvalidImage(f"不支援的檔案格式：{ext or '(無副檔名)'}")
    sha1 = hashlib.sha1(data).hexdigest()
    img = _decode(ext, data)

    orientation = img.getexif().get(0x0112, 1) if hasattr(img, "getexif") else 1
    needs_rewrite = orientation not in (1, None) or ext in CONVERT_TO_PNG or ext in CONVERT_TO_JPG

    if ext in CONVERT_TO_PNG:
        out_ext = ".png"
    elif ext in CONVERT_TO_JPG or ext == ".jpeg":
        out_ext = ".jpg"
    else:
        out_ext = ext
    stored_name = f"{uuid.uuid4().hex}{out_ext}"
    dest = image_path(project_id, stored_name)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if needs_rewrite:
        img = ImageOps.exif_transpose(img)
        if out_ext == ".jpg" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        save_kwargs = {"quality": 95} if out_ext == ".jpg" else {}
        img.save(dest, **save_kwargs)
    else:
        dest.write_bytes(data)

    make_thumbnail(img, thumb_path(project_id, stored_name))
    return StoredImage(stored_name=stored_name, width=img.width, height=img.height, sha1=sha1)


def _decode(ext: str, data: bytes) -> PILImage.Image:
    try:
        if ext in RAW_EXTS:
            return _develop_raw(data)
        if ext in HEIF_EXTS:
            _register_heif()
        img = PILImage.open(io.BytesIO(data))
        img.load()
        return img
    except InvalidImage:
        raise
    except Exception as exc:  # decoders raise their own error types (LibRawError, HeifError ...)
        raise InvalidImage(f"無法讀取圖片：{exc}") from exc


def _register_heif() -> None:
    try:
        import pillow_heif
    except ImportError as exc:
        raise InvalidImage("要上傳 HEIC 需要安裝 pillow-heif：pip install pillow-heif") from exc
    pillow_heif.register_heif_opener()  # idempotent


def _develop_raw(data: bytes) -> PILImage.Image:
    """Render a RAW file the way the camera would: camera white balance, sRGB, 8 bit.

    LibRaw applies the orientation flag itself, so the result is already upright.
    """
    try:
        import rawpy
    except ImportError as exc:
        raise InvalidImage("要上傳 DNG 需要安裝 rawpy：pip install rawpy") from exc
    with rawpy.imread(io.BytesIO(data)) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
    return PILImage.fromarray(rgb)


def make_thumbnail(img: PILImage.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb = img.copy()
    if thumb.mode not in ("RGB", "L"):
        thumb = thumb.convert("RGB")
    thumb.thumbnail((settings.thumbnail_size, settings.thumbnail_size))
    thumb.save(dest, "JPEG", quality=82)


def delete_image_files(project_id: int, stored_name: str) -> None:
    for p in (image_path(project_id, stored_name), thumb_path(project_id, stored_name)):
        p.unlink(missing_ok=True)


def delete_project_files(project_id: int) -> None:
    shutil.rmtree(project_dir(project_id), ignore_errors=True)
