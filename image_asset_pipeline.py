#!/usr/bin/env python3
"""
Image Asset Pipeline
====================
AI-powered image categorization, renaming, and indexing.

Commands:
  organize  - Sort images into category folders using OCR + AI Vision
  rename    - Bulk rename files with metadata-rich names
  index     - Export a CSV/JSON asset manifest
  dedup     - Find duplicate files by checksum or name

Works with Google Drive folders or local directories.

Pipeline:
  1. organize  -> Sort IMG_4530.jpg into "Blue Mountain Blend/" folder
  2. rename    -> Rename to "Blue-Mountain-Blend_product-shot_001.jpg"
  3. index     -> Export spreadsheet with every file, category, confidence, link
"""

import os
import io
import json
import base64
import re
import csv
import sys
import time
import mimetypes
import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

CONFIDENCE_THRESHOLD = 0.60
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.tiff', '.bmp'}
MAX_IMAGE_SIZE_BYTES = 3 * 1024 * 1024
MAX_IMAGE_DIMENSION = 6000
DEFAULT_BATCH_SIZE = 50

DEFAULT_STYLE_CATEGORIES = [
    "Product Shot",
    "White Background",
    "Lifestyle",
    "Model"
]


class MatchMethod(Enum):
    FILENAME = "filename"
    OCR = "ocr"
    VISION = "vision"
    FOLDER = "folder"
    UNMATCHED = "unmatched"


@dataclass
class MatchResult:
    category: str
    confidence: float
    method: MatchMethod
    details: str = ""
    detected_text: str = ""
    style: str = ""


@dataclass
class AssetRecord:
    """A single asset in the index."""
    filename: str
    original_filename: str = ""
    category: str = ""
    style: str = ""
    confidence: float = 0.0
    method: str = ""
    detected_text: str = ""
    file_path: str = ""
    file_id: str = ""
    drive_link: str = ""
    md5: str = ""
    size_bytes: int = 0


# ============================================================================
# Image Processing
# ============================================================================

def resize_image_for_api(image_bytes: bytes) -> Tuple[bytes, str]:
    """Resize and compress image for API limits."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        w, h = img.size
        if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
            ratio = min(MAX_IMAGE_DIMENSION / w, MAX_IMAGE_DIMENSION / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        quality = 85
        while quality >= 20:
            buf.seek(0)
            buf.truncate()
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            if buf.tell() <= MAX_IMAGE_SIZE_BYTES:
                break
            quality -= 10

        while buf.tell() > MAX_IMAGE_SIZE_BYTES and img.size[0] > 500:
            img = img.resize((int(img.size[0] * 0.7), int(img.size[1] * 0.7)), Image.Resampling.LANCZOS)
            buf.seek(0)
            buf.truncate()
            img.save(buf, format='JPEG', quality=60, optimize=True)

        buf.seek(0)
        return buf.read(), 'image/jpeg'
    except Exception as e:
        print(f"    [Image processing error: {e}]")
        return image_bytes, 'image/jpeg'


def needs_resize(image_bytes: bytes) -> bool:
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        return True
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        return w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION
    except:
        return False


# ============================================================================
# Category Catalog
# ============================================================================

class CategoryCatalog:
    """Manages categories with keywords and label text mappings."""

    def __init__(self, catalog_path: str = "categories.json"):
        self.categories = []
        self.label_text_map = {}
        self.style_categories = DEFAULT_STYLE_CATEGORIES

        if os.path.exists(catalog_path):
            with open(catalog_path, 'r') as f:
                data = json.load(f)
            self.categories = data.get('categories', data.get('products', []))
            self.label_text_map = data.get('label_text_map', {})
            if 'style_categories' in data:
                self.style_categories = data['style_categories']
        else:
            print(f"  [!] No catalog at {catalog_path} - using OCR/Vision only")

    def match_filename(self, filename: str) -> Optional[MatchResult]:
        name_clean = re.sub(r'[_\-\.]', ' ', Path(filename.lower()).stem)

        best_match = None
        best_score = 0

        for cat in self.categories:
            score = 0
            for kw in cat.get('keywords', []):
                if kw.lower() in name_clean:
                    score += len(kw.split())
            if score > best_score:
                best_score = score
                best_match = cat

        if best_match and best_score > 0:
            return MatchResult(
                category=best_match['title'],
                confidence=min(0.6 + best_score * 0.1, 0.95),
                method=MatchMethod.FILENAME,
                details="Matched keywords in filename"
            )
        return None

    def match_ocr_text(self, ocr_text: str) -> Tuple[str, float]:
        if not ocr_text or not self.label_text_map:
            return "Unknown", 0.0

        text_upper = ocr_text.upper()
        text_clean = re.sub(r'[^A-Z0-9\s-]', '', text_upper)

        for label, name in self.label_text_map.items():
            if label.upper() in text_upper:
                return name, 0.95

        for label, name in self.label_text_map.items():
            label_clean = re.sub(r'[^A-Z0-9\s-]', '', label.upper())
            if label_clean in text_clean:
                return name, 0.90

        best_match, best_score = None, 0
        for label, name in self.label_text_map.items():
            lw = set(label.upper().split())
            tw = set(text_clean.split())
            overlap = lw & tw
            if overlap:
                score = len(overlap) / len(lw)
                if score > best_score:
                    best_score = score
                    best_match = name

        if best_match and best_score >= 0.5:
            return best_match, 0.70 + best_score * 0.2

        return "Unknown", 0.0

    def get_category_list(self) -> list:
        return [c['title'] for c in self.categories]


# ============================================================================
# OCR Engine (Google Cloud Vision - optional)
# ============================================================================

class OCREngine:
    """Google Cloud Vision OCR. Gracefully disabled if not available."""

    def __init__(self):
        self.client = None
        try:
            from google.cloud import vision
            self.vision = vision
            self.client = vision.ImageAnnotatorClient()
            print("  [+] Cloud Vision OCR ready")
        except Exception:
            print("  [.] Cloud Vision unavailable (install google-cloud-vision to enable)")

    def extract_text(self, image_bytes: bytes) -> Tuple[str, float]:
        if not self.client:
            return "", 0.0
        try:
            image = self.vision.Image(content=image_bytes)
            response = self.client.text_detection(image=image)
            if response.error.message:
                return "", 0.0
            texts = response.text_annotations
            if not texts:
                return "", 0.0
            return texts[0].description.strip(), 0.9
        except Exception as e:
            print(f"    OCR error: {e}")
            return "", 0.0


# ============================================================================
# AI Vision Engine (Claude)
# ============================================================================

class VisionEngine:
    """Claude Vision for image analysis and style classification."""

    def __init__(self, category_list: list, custom_prompt: str = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.category_list = category_list
        self.custom_prompt = custom_prompt

    def identify(self, image_bytes: bytes, filename: str, mime_type: str) -> MatchResult:
        if needs_resize(image_bytes):
            image_bytes, mime_type = resize_image_for_api(image_bytes)

        prompt = self.custom_prompt or self._build_prompt()
        b64 = base64.b64encode(image_bytes).decode('utf-8')

        try:
            resp = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            text = resp.content[0].text
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                result = json.loads(m.group())
                guess = result.get('category', result.get('product_guess', 'Unknown'))
                conf = float(result.get('confidence', 0.0))

                if guess not in self.category_list and guess != "Unknown":
                    for valid in self.category_list:
                        if guess.lower() in valid.lower():
                            guess = valid
                            break
                    else:
                        guess, conf = "Unknown", 0.4

                if guess.lower() == "unknown":
                    conf = min(conf, 0.50)

                return MatchResult(
                    category=guess, confidence=conf,
                    method=MatchMethod.VISION,
                    details=result.get('visible_clues', '')[:80]
                )
        except Exception as e:
            print(f"    Vision error: {e}")

        return MatchResult(category="Unknown", confidence=0.0,
                           method=MatchMethod.UNMATCHED, details="Vision failed")

    def classify_style(self, image_bytes: bytes, mime_type: str,
                       style_categories: list) -> str:
        if needs_resize(image_bytes):
            image_bytes, mime_type = resize_image_for_api(image_bytes)

        cats = "\n".join(f'{i+1}. "{c}"' for i, c in enumerate(style_categories))
        prompt = f"Classify this image into ONE category:\n\n{cats}\n\nRESPOND WITH ONLY the category name."
        b64 = base64.b64encode(image_bytes).decode('utf-8')

        try:
            resp = self.client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=50,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}},
                    {"type": "text", "text": prompt}
                ]}]
            )
            answer = resp.content[0].text.strip()
            for cat in style_categories:
                if cat.lower() in answer.lower():
                    return cat
            return style_categories[0]
        except:
            return style_categories[0]

    def _build_prompt(self):
        cats = "\n".join(f"- {c}" for c in self.category_list)
        return f"""Identify the category of this image from this list:

{cats}

RESPOND WITH JSON:
{{
    "category": "Category name from list, or Unknown",
    "confidence": 0.0 to 1.0,
    "visible_clues": "What you see that led to your guess"
}}

If you cannot confidently identify, use "Unknown"."""


# ============================================================================
# Storage Backends
# ============================================================================

class LocalStorage:
    """Local filesystem operations."""

    def __init__(self, base_path: str):
        self.base = Path(base_path)

    def list_images(self, folder_path: str = "") -> List[Dict]:
        target = self.base / folder_path if folder_path else self.base
        files = []
        for f in sorted(target.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append({
                    'id': str(f),
                    'name': f.name,
                    'path': str(f),
                    'mimeType': mimetypes.guess_type(f.name)[0] or 'image/jpeg'
                })
        return files

    def read_file(self, file_path: str) -> bytes:
        return Path(file_path).read_bytes()

    def create_folder(self, name: str, parent: str = "") -> str:
        target = Path(parent) / name if parent else self.base / name
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def move_file(self, file_path: str, dest_folder: str):
        src = Path(file_path)
        dst = Path(dest_folder) / src.name
        if dst.exists():
            stem, ext = dst.stem, dst.suffix
            i = 1
            while dst.exists():
                dst = Path(dest_folder) / f"{stem}_{i}{ext}"
                i += 1
        shutil.move(str(src), str(dst))
        return str(dst)

    def rename_file(self, file_path: str, new_name: str) -> str:
        src = Path(file_path)
        dst = src.parent / new_name
        if dst.exists() and dst != src:
            stem, ext = Path(new_name).stem, Path(new_name).suffix
            i = 1
            while dst.exists():
                dst = src.parent / f"{stem}_{i}{ext}"
                i += 1
        src.rename(dst)
        return str(dst)

    def list_all_recursive(self, folder_path: str = "") -> List[Dict]:
        target = Path(folder_path) if folder_path else self.base
        files = []
        for f in sorted(target.rglob('*')):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                rel = f.relative_to(target)
                files.append({
                    'id': str(f),
                    'name': f.name,
                    'path': str(f),
                    'folder': str(rel.parent) if str(rel.parent) != '.' else '',
                    'size': f.stat().st_size,
                    'mimeType': mimetypes.guess_type(f.name)[0] or 'image/jpeg'
                })
        return files

    def file_md5(self, file_path: str) -> str:
        h = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()


class DriveStorage:
    """Google Drive operations."""

    def __init__(self, credentials_path="credentials.json", token_path="token.json"):
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        self._dl_class = MediaIoBaseDownload
        scopes = ['https://www.googleapis.com/auth/drive']

        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(f"Missing {credentials_path}")
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
                creds = flow.run_local_server(port=0)
            with open(token_path, 'w') as t:
                t.write(creds.to_json())

        self.service = build('drive', 'v3', credentials=creds)
        self._cache = {}

    def list_images(self, folder_id: str) -> List[Dict]:
        q = f"'{folder_id}' in parents and trashed=false"
        files, token = [], None
        while True:
            r = self.service.files().list(
                q=q, spaces='drive', pageToken=token, pageSize=100,
                fields='nextPageToken, files(id, name, mimeType)',
                supportsAllDrives=True, includeItemsFromAllDrives=True
            ).execute()
            for f in r.get('files', []):
                ext = Path(f['name']).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS or f['mimeType'].startswith('image/'):
                    files.append(f)
            token = r.get('nextPageToken')
            if not token:
                break
        return files

    def read_file(self, file_id: str) -> bytes:
        req = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        buf = io.BytesIO()
        dl = self._dl_class(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        buf.seek(0)
        return buf.getvalue()

    def create_folder(self, name: str, parent_id: str) -> str:
        key = f"{parent_id}/{name}"
        if key in self._cache:
            return self._cache[key]

        q = (f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
             f"and '{parent_id}' in parents and trashed=false")
        r = self.service.files().list(
            q=q, spaces='drive', fields='files(id)',
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()

        if r.get('files'):
            self._cache[key] = r['files'][0]['id']
            return self._cache[key]

        meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]}
        f = self.service.files().create(body=meta, fields='id',
                                        supportsAllDrives=True).execute()
        self._cache[key] = f['id']
        return f['id']

    def move_file(self, file_id: str, new_folder_id: str):
        f = self.service.files().get(fileId=file_id, fields='parents',
                                     supportsAllDrives=True).execute()
        prev = ",".join(f.get('parents', []))
        self.service.files().update(
            fileId=file_id, addParents=new_folder_id,
            removeParents=prev, fields='id', supportsAllDrives=True
        ).execute()

    def rename_file(self, file_id: str, new_name: str):
        self.service.files().update(
            fileId=file_id, body={'name': new_name},
            fields='id', supportsAllDrives=True
        ).execute()

    def get_file_info(self, file_id: str) -> Dict:
        return self.service.files().get(
            fileId=file_id,
            fields='id, name, mimeType, md5Checksum, size, webViewLink, parents',
            supportsAllDrives=True
        ).execute()

    def list_all_recursive(self, folder_id: str, path="") -> List[Dict]:
        files = []
        try:
            if not path:
                f = self.service.files().get(fileId=folder_id, fields='name',
                                             supportsAllDrives=True).execute()
                path = f.get('name', 'Root')

            q = f"'{folder_id}' in parents and trashed=false"
            token = None
            while True:
                r = self.service.files().list(
                    q=q, spaces='drive', pageToken=token, pageSize=1000,
                    fields='nextPageToken, files(id, name, mimeType, md5Checksum, size, webViewLink)',
                    supportsAllDrives=True, includeItemsFromAllDrives=True
                ).execute()
                for item in r.get('files', []):
                    if item['mimeType'] == 'application/vnd.google-apps.folder':
                        files.extend(self.list_all_recursive(item['id'], f"{path}/{item['name']}"))
                    else:
                        ext = Path(item['name']).suffix.lower()
                        if ext in SUPPORTED_EXTENSIONS or item['mimeType'].startswith('image/'):
                            files.append({
                                'id': item['id'], 'name': item['name'],
                                'path': path, 'folder': path.split('/')[-1] if '/' in path else '',
                                'size': int(item.get('size', 0)),
                                'md5': item.get('md5Checksum', ''),
                                'link': item.get('webViewLink', ''),
                                'mimeType': item['mimeType']
                            })
                token = r.get('nextPageToken')
                if not token:
                    break
        except Exception as e:
            print(f"  Error scanning: {e}")
        return files


# ============================================================================
# Command: organize
# ============================================================================

def cmd_organize(args):
    """Sort images into category folders using AI."""
    print("=" * 60)
    print("Image Asset Pipeline - ORGANIZE")
    print("=" * 60)

    catalog = CategoryCatalog(args.catalog)

    if args.local:
        storage = LocalStorage(args.source)
        print(f"  [+] Local mode: {args.source}")
    else:
        storage = DriveStorage(args.credentials)
        print("  [+] Google Drive connected")

    ocr = OCREngine()

    import anthropic
    vision = VisionEngine(catalog.get_category_list())
    print("  [+] Claude Vision ready")

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence

    source = args.source
    print(f"\nScanning: {source}")
    images = storage.list_images(source)
    total = len(images)
    print(f"Found {total} images\n")

    if total == 0:
        print("No images found!")
        return

    # Setup output folders
    if args.local:
        base_output = storage.create_folder("Organized" if not args.in_place else "", source)
        if args.in_place:
            base_output = source
        else:
            base_output = storage.create_folder("Organized", "")
            base_output = str(Path(source) / "Organized")
            Path(base_output).mkdir(exist_ok=True)
        review_dir = storage.create_folder("_Review", base_output)
    else:
        if args.in_place:
            base_output = source
        else:
            base_output = storage.create_folder("Organized", source)
        review_dir = storage.create_folder("_Review", base_output)

    # Batching
    if args.batch:
        start_idx = (args.start_batch - 1) * args.batch
        end_idx = min(start_idx + args.batch, total)
        if start_idx >= total:
            print(f"Batch exceeds total images.")
            return
        images = images[start_idx:end_idx]
        print(f"Batch {args.start_batch}: images {start_idx+1}-{end_idx} of {total}\n")

    stats = defaultdict(int)
    stats['total'] = len(images)
    records = []

    for i, img in enumerate(sorted(images, key=lambda x: x['name']), 1):
        fname = img['name']
        fid = img['id'] if not args.local else img['path']
        mime = img.get('mimeType', 'image/jpeg')

        print(f"\n[{i}/{len(images)}] {fname}")

        match = None

        # Step 1: filename
        match = catalog.match_filename(fname)
        if match and match.confidence >= CONFIDENCE_THRESHOLD:
            pass
        else:
            # Step 2: download
            print(f"  -> Analyzing...")
            image_bytes = storage.read_file(fid)

            # Step 3: OCR
            ocr_text, _ = ocr.extract_text(image_bytes)
            if ocr_text:
                display = ocr_text[:50] + "..." if len(ocr_text) > 50 else ocr_text
                print(f'    OCR: "{display}"')
                product, conf = catalog.match_ocr_text(ocr_text)
                if product != "Unknown" and conf >= CONFIDENCE_THRESHOLD:
                    match = MatchResult(category=product, confidence=conf,
                                        method=MatchMethod.OCR,
                                        details=f"Label: '{ocr_text[:40]}'",
                                        detected_text=ocr_text)

            # Step 4: Claude Vision
            if not match or match.confidence < CONFIDENCE_THRESHOLD:
                print(f"  -> Vision analysis...")
                match = vision.identify(image_bytes, fname, mime)

        # Move file
        if match and match.confidence >= CONFIDENCE_THRESHOLD and match.category != "Unknown":
            print(f"  -> {match.category} ({match.confidence:.0%}) [{match.method.value}]")
            dest = storage.create_folder(match.category, base_output)
            if args.local:
                storage.move_file(fid, dest)
            else:
                storage.move_file(fid, dest)
            stats[f'{match.method.value}_matched'] += 1
        else:
            cat = match.category if match else "Unknown"
            conf = match.confidence if match else 0.0
            print(f"  ? {cat} ({conf:.0%}) -> _Review")
            if args.local:
                storage.move_file(fid, review_dir)
            else:
                storage.move_file(fid, review_dir)
            stats['review'] += 1
            match = match or MatchResult(category="Unknown", confidence=0.0,
                                          method=MatchMethod.UNMATCHED)

        records.append(AssetRecord(
            filename=fname, category=match.category,
            confidence=match.confidence, method=match.method.value,
            detected_text=match.detected_text
        ))

    # Summary
    print("\n" + "=" * 60)
    print("ORGANIZE COMPLETE")
    print("=" * 60)
    print(f"Total:       {stats['total']}")
    print(f"Filename:    {stats.get('filename_matched', 0)}")
    print(f"OCR:         {stats.get('ocr_matched', 0)}")
    print(f"Vision:      {stats.get('vision_matched', 0)}")
    print(f"Review:      {stats.get('review', 0)}")

    success = sum(stats.get(f'{m}_matched', 0) for m in ['filename', 'ocr', 'vision'])
    if stats['total'] > 0:
        print(f"Success:     {success}/{stats['total']} ({100*success/stats['total']:.1f}%)")

    # Save organize log
    if args.dry_run:
        print("\n[DRY RUN] No files were moved.")
    else:
        log = f"organize_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(log, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['filename', 'category', 'confidence', 'method', 'detected_text'])
            w.writeheader()
            for r in records:
                w.writerow({'filename': r.filename, 'category': r.category,
                            'confidence': f"{r.confidence:.2f}", 'method': r.method,
                            'detected_text': r.detected_text[:100]})
        print(f"\nLog: {log}")


# ============================================================================
# Command: rename
# ============================================================================

def slugify(text: str) -> str:
    """Convert text to filename-safe slug."""
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text.strip())
    return text


def cmd_rename(args):
    """Bulk rename files with metadata-rich names based on folder structure."""
    print("=" * 60)
    print("Image Asset Pipeline - RENAME")
    print("=" * 60)

    template = args.template  # e.g. "{category}_{style}_{seq}"

    if args.local:
        storage = LocalStorage(args.source)
        print(f"  Source: {args.source}")

        # Walk organized folder structure
        base = Path(args.source)
        renames = []

        for category_dir in sorted(base.iterdir()):
            if not category_dir.is_dir():
                continue
            if category_dir.name.startswith('_'):
                continue  # skip _Review etc

            category = category_dir.name
            images = sorted(f for f in category_dir.iterdir()
                            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)

            # Check for style sub-folders
            style_dirs = [d for d in category_dir.iterdir() if d.is_dir()]
            if style_dirs:
                for style_dir in sorted(style_dirs):
                    style = style_dir.name
                    style_images = sorted(f for f in style_dir.iterdir()
                                          if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
                    for seq, img in enumerate(style_images, 1):
                        new_name = _build_name(template, category, style, seq, img.suffix)
                        renames.append((str(img), new_name, category, style))
            else:
                for seq, img in enumerate(images, 1):
                    new_name = _build_name(template, category, "", seq, img.suffix)
                    renames.append((str(img), new_name, category, ""))

    else:
        storage = DriveStorage(args.credentials)
        print("  [+] Google Drive connected")

        # Walk folder structure on Drive
        renames = []
        source = args.source

        q = f"'{source}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'"
        r = storage.service.files().list(
            q=q, spaces='drive', fields='files(id, name)',
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()

        for cat_folder in sorted(r.get('files', []), key=lambda x: x['name']):
            if cat_folder['name'].startswith('_'):
                continue

            category = cat_folder['name']
            images = storage.list_images(cat_folder['id'])
            images.sort(key=lambda x: x['name'])

            # Check for sub-folders (styles)
            subq = (f"'{cat_folder['id']}' in parents and trashed=false "
                    f"and mimeType='application/vnd.google-apps.folder'")
            subr = storage.service.files().list(
                q=subq, spaces='drive', fields='files(id, name)',
                supportsAllDrives=True, includeItemsFromAllDrives=True
            ).execute()
            sub_folders = subr.get('files', [])

            if sub_folders:
                for sf in sorted(sub_folders, key=lambda x: x['name']):
                    style = sf['name']
                    style_imgs = storage.list_images(sf['id'])
                    style_imgs.sort(key=lambda x: x['name'])
                    for seq, img in enumerate(style_imgs, 1):
                        ext = Path(img['name']).suffix
                        new_name = _build_name(template, category, style, seq, ext)
                        renames.append((img['id'], new_name, category, style))
            else:
                for seq, img in enumerate(images, 1):
                    ext = Path(img['name']).suffix
                    new_name = _build_name(template, category, "", seq, ext)
                    renames.append((img['id'], new_name, category, ""))

    print(f"\n{len(renames)} files to rename\n")

    if not renames:
        print("Nothing to rename.")
        return

    # Preview
    print("Preview (first 10):")
    for fid, new_name, cat, style in renames[:10]:
        old = Path(fid).name if args.local else fid[:12] + "..."
        print(f"  {old:40} -> {new_name}")
    if len(renames) > 10:
        print(f"  ... and {len(renames) - 10} more\n")

    if args.dry_run:
        print("[DRY RUN] No files renamed.")
        # Still save the plan
        plan = f"rename_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(plan, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['file_id_or_path', 'new_name', 'category', 'style'])
            for row in renames:
                w.writerow(row)
        print(f"Plan saved: {plan}")
        return

    # Execute renames
    success, errors = 0, 0
    for fid, new_name, cat, style in renames:
        try:
            if args.local:
                storage.rename_file(fid, new_name)
            else:
                storage.rename_file(fid, new_name)
            success += 1
        except Exception as e:
            print(f"  [X] Error renaming {fid}: {e}")
            errors += 1

    print(f"\nRenamed: {success}  Errors: {errors}")

    # Save log
    log = f"rename_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['original', 'new_name', 'category', 'style'])
        for row in renames:
            w.writerow(row)
    print(f"Log: {log}")


def _build_name(template: str, category: str, style: str, seq: int, ext: str) -> str:
    """Build a filename from template."""
    name = template.format(
        category=slugify(category),
        style=slugify(style) if style else "",
        seq=f"{seq:03d}",
        date=datetime.now().strftime('%Y%m%d')
    )
    # Clean up repeated separators from empty tokens
    name = re.sub(r'[-_]{2,}', '_', name).strip('-_')
    return name + ext.lower()


# ============================================================================
# Command: index
# ============================================================================

def cmd_index(args):
    """Generate a CSV/JSON manifest of all images."""
    print("=" * 60)
    print("Image Asset Pipeline - INDEX")
    print("=" * 60)

    if args.local:
        storage = LocalStorage(args.source)
        print(f"  Scanning: {args.source}")
        files = storage.list_all_recursive(args.source)

        records = []
        for f in files:
            md5 = ""
            if args.checksums:
                md5 = storage.file_md5(f['path'])

            records.append({
                'filename': f['name'],
                'category': f.get('folder', ''),
                'path': f['path'],
                'size_bytes': f.get('size', 0),
                'md5': md5,
                'mime_type': f.get('mimeType', '')
            })
    else:
        storage = DriveStorage(args.credentials)
        print("  [+] Google Drive connected")
        print(f"  Scanning: {args.source}")
        files = storage.list_all_recursive(args.source)

        records = []
        for f in files:
            records.append({
                'filename': f['name'],
                'category': f.get('folder', ''),
                'path': f.get('path', ''),
                'file_id': f.get('id', ''),
                'drive_link': f.get('link', ''),
                'size_bytes': f.get('size', 0),
                'md5': f.get('md5', ''),
                'mime_type': f.get('mimeType', '')
            })

    print(f"\n  Found {len(records)} images")

    # Category summary
    by_cat = defaultdict(int)
    for r in records:
        by_cat[r['category'] or '(root)'] += 1
    print(f"\n  Categories:")
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        bar = "#" * min(count // 2, 30)
        print(f"    {cat:40} {count:5}  {bar}")

    # Export
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.format == 'csv' or args.format == 'both':
        csv_path = args.output or f"asset_index_{ts}.csv"
        if not csv_path.endswith('.csv'):
            csv_path += '.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else [])
            w.writeheader()
            w.writerows(records)
        print(f"\n  CSV: {csv_path}")

    if args.format == 'json' or args.format == 'both':
        json_path = args.output or f"asset_index_{ts}.json"
        if not json_path.endswith('.json'):
            json_path += '.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'generated': ts, 'total': len(records),
                       'summary': dict(by_cat), 'assets': records}, f, indent=2)
        print(f"  JSON: {json_path}")

    print(f"\n  Total: {len(records)} assets indexed")


# ============================================================================
# Command: dedup
# ============================================================================

def cmd_dedup(args):
    """Find duplicate files by checksum or filename."""
    print("=" * 60)
    print("Image Asset Pipeline - DEDUP")
    print("=" * 60)

    if args.local:
        storage = LocalStorage(args.source)
        print(f"  Scanning: {args.source}")
        files = storage.list_all_recursive(args.source)

        # Compute checksums
        print(f"  Computing checksums for {len(files)} files...")
        for f in files:
            f['md5'] = storage.file_md5(f['path'])
            f['size'] = f.get('size', Path(f['path']).stat().st_size)
    else:
        storage = DriveStorage(args.credentials)
        print("  [+] Google Drive connected")
        print(f"  Scanning: {args.source}")
        files = storage.list_all_recursive(args.source)

    print(f"  Found {len(files)} files\n")

    # Exact duplicates (by checksum)
    by_md5 = defaultdict(list)
    for f in files:
        if f.get('md5'):
            by_md5[f['md5']].append(f)
    exact_dupes = {k: v for k, v in by_md5.items() if len(v) > 1}

    print(f"EXACT DUPLICATES (same content): {len(exact_dupes)} sets")
    print("-" * 60)
    total_wasted = 0
    for md5, items in exact_dupes.items():
        size = items[0].get('size', 0)
        wasted = size * (len(items) - 1)
        total_wasted += wasted
        print(f"\n  {items[0]['name']} ({_fmt_size(size)}) x {len(items)} copies")
        for item in items:
            loc = item.get('path', item.get('folder', ''))
            print(f"    - {loc}/{item['name']}")

    if exact_dupes:
        print(f"\n  Wasted space: {_fmt_size(total_wasted)}")
    else:
        print("  None found.")

    # Same-name files
    by_name = defaultdict(list)
    for f in files:
        by_name[f['name'].lower()].append(f)
    name_dupes = {k: v for k, v in by_name.items() if len(v) > 1}

    # Filter out ones we already reported
    exact_md5s = set(exact_dupes.keys())
    name_only = {}
    for name, items in name_dupes.items():
        md5s = set(f.get('md5', '') for f in items)
        if len(md5s) > 1:
            name_only[name] = items

    print(f"\n\nSAME FILENAME, DIFFERENT CONTENT: {len(name_only)} sets")
    print("-" * 60)
    for name, items in name_only.items():
        print(f"\n  {name} - {len(items)} versions")
        for item in items:
            loc = item.get('path', item.get('folder', ''))
            size = _fmt_size(item.get('size', 0))
            print(f"    - {loc} ({size})")

    if not name_only:
        print("  None found.")

    # Save report
    report = f"dedup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(report, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['type', 'filename', 'location', 'size', 'md5'])
        for md5, items in exact_dupes.items():
            for item in items:
                w.writerow(['exact_duplicate', item['name'],
                            item.get('path', ''), item.get('size', 0), md5])
        for name, items in name_only.items():
            for item in items:
                w.writerow(['same_name', item['name'],
                            item.get('path', ''), item.get('size', 0),
                            item.get('md5', '')])
    print(f"\nReport: {report}")


def _fmt_size(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='image-asset-pipeline',
        description='AI-powered image categorization, renaming, and indexing'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # -- organize --
    org = sub.add_parser('organize', help='Sort images into category folders')
    org.add_argument('source', help='Local path or Google Drive folder ID')
    org.add_argument('--local', action='store_true', help='Source is a local directory')
    org.add_argument('--catalog', default='categories.json', help='Category catalog JSON')
    org.add_argument('--credentials', default='credentials.json', help='Google OAuth credentials')
    org.add_argument('--in-place', action='store_true', help='Organize in place')
    org.add_argument('--confidence', type=float, default=0.60, help='Confidence threshold')
    org.add_argument('--batch', type=int, help='Batch size')
    org.add_argument('--start-batch', type=int, default=1, help='Start from batch N')
    org.add_argument('--dry-run', action='store_true', help='Preview without moving files')

    # -- rename --
    ren = sub.add_parser('rename', help='Bulk rename files with metadata-rich names')
    ren.add_argument('source', help='Organized folder (local path or Drive ID)')
    ren.add_argument('--local', action='store_true', help='Source is local')
    ren.add_argument('--credentials', default='credentials.json', help='Google OAuth credentials')
    ren.add_argument('--template', default='{category}_{style}_{seq}',
                     help='Naming template. Tokens: {category}, {style}, {seq}, {date}')
    ren.add_argument('--dry-run', action='store_true', help='Preview without renaming')

    # -- index --
    idx = sub.add_parser('index', help='Export asset manifest (CSV/JSON)')
    idx.add_argument('source', help='Folder to index (local path or Drive ID)')
    idx.add_argument('--local', action='store_true', help='Source is local')
    idx.add_argument('--credentials', default='credentials.json', help='Google OAuth credentials')
    idx.add_argument('--format', choices=['csv', 'json', 'both'], default='csv', help='Output format')
    idx.add_argument('--output', help='Output file path')
    idx.add_argument('--checksums', action='store_true', help='Compute MD5 checksums (slower)')

    # -- dedup --
    dup = sub.add_parser('dedup', help='Find duplicate images')
    dup.add_argument('source', help='Folder to scan (local path or Drive ID)')
    dup.add_argument('--local', action='store_true', help='Source is local')
    dup.add_argument('--credentials', default='credentials.json', help='Google OAuth credentials')

    args = parser.parse_args()

    if args.command == 'organize':
        cmd_organize(args)
    elif args.command == 'rename':
        cmd_rename(args)
    elif args.command == 'index':
        cmd_index(args)
    elif args.command == 'dedup':
        cmd_dedup(args)


if __name__ == '__main__':
    main()
