#!/usr/bin/env python3
"""
Google Drive Photo Organizer
=============================
AI-powered photo organization with hybrid OCR + Vision approach.

Pipeline:
1. Filename matching (fast, high confidence)
2. Google Cloud Vision OCR (reads label text accurately)
3. Claude Vision fallback (for visual/contextual detection)
4. Style classification within product folders (optional Phase 2)

Requires:
- Google Cloud project with Vision API enabled
- Anthropic API key (ANTHROPIC_API_KEY env var)
- credentials.json for Google Drive OAuth
- products.json product catalog
"""

import os
import io
import json
import base64
import re
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# Google APIs
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Google Cloud Vision
from google.cloud import vision
from google.oauth2 import service_account

# Anthropic for fallback vision
import anthropic

# Image processing
from PIL import Image

# ============================================================================
# Configuration
# ============================================================================

SCOPES = ['https://www.googleapis.com/auth/drive']
CONFIDENCE_THRESHOLD = 0.60
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.tiff'}

# Image size limits
MAX_IMAGE_SIZE_BYTES = 3 * 1024 * 1024  # 3MB
MAX_IMAGE_DIMENSION = 6000  # pixels

# Default batch size
DEFAULT_BATCH_SIZE = 50

# Style categories for Phase 2
STYLE_CATEGORIES = [
    "Product Shot",
    "White Background",
    "Lifestyle",
    "Model"
]


class MatchMethod(Enum):
    FILENAME = "filename"
    OCR = "ocr"
    VISION = "vision"
    UNMATCHED = "unmatched"


@dataclass
class MatchResult:
    product_title: str
    confidence: float
    method: MatchMethod
    details: str = ""
    multiple_products: bool = False
    products_found: list = None
    detected_text: str = ""
    style: str = ""


# ============================================================================
# Image Processing Utilities
# ============================================================================

def resize_image_for_api(image_bytes: bytes, filename: str) -> Tuple[bytes, str]:
    """Resize and compress image for API limits."""
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if needed
        width, height = img.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            ratio = min(MAX_IMAGE_DIMENSION / width, MAX_IMAGE_DIMENSION / height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"    [Resized from {width}x{height} to {new_size[0]}x{new_size[1]}]")

        # Compress
        buffer = io.BytesIO()
        quality = 85

        while quality >= 20:
            buffer.seek(0)
            buffer.truncate()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            if buffer.tell() <= MAX_IMAGE_SIZE_BYTES:
                break
            quality -= 10

        while buffer.tell() > MAX_IMAGE_SIZE_BYTES and img.size[0] > 500:
            new_size = (int(img.size[0] * 0.7), int(img.size[1] * 0.7))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            buffer.seek(0)
            buffer.truncate()
            img.save(buffer, format='JPEG', quality=60, optimize=True)

        buffer.seek(0)
        return buffer.read(), 'image/jpeg'

    except Exception as e:
        print(f"    [Image processing error: {e}]")
        return image_bytes, 'image/jpeg'


def needs_processing(image_bytes: bytes) -> bool:
    """Check if image needs resizing."""
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        return True
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        return width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION
    except:
        return False


# ============================================================================
# Google Cloud Vision OCR
# ============================================================================

class CloudVisionOCR:
    """Google Cloud Vision API for accurate text extraction."""

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            self.client = vision.ImageAnnotatorClient()
            print("  [+] Cloud Vision API initialized")
        except Exception as e:
            print(f"  [!] Cloud Vision init warning: {e}")
            try:
                self.client = vision.ImageAnnotatorClient()
            except Exception:
                print("  [!] Cloud Vision unavailable - will rely on filename + Claude Vision")

    def extract_text(self, image_bytes: bytes) -> Tuple[str, float]:
        """Extract text from image. Returns (text, confidence)."""
        if not self.client:
            return "", 0.0

        try:
            image = vision.Image(content=image_bytes)
            response = self.client.text_detection(image=image)

            if response.error.message:
                print(f"    OCR Error: {response.error.message}")
                return "", 0.0

            texts = response.text_annotations
            if not texts:
                return "", 0.0

            full_text = texts[0].description.strip()
            confidence = 0.9
            if len(texts) > 1:
                confidences = []
                for text in texts[1:]:
                    if hasattr(text, 'confidence'):
                        confidences.append(text.confidence)
                if confidences:
                    confidence = sum(confidences) / len(confidences)

            return full_text, confidence

        except Exception as e:
            print(f"    OCR Exception: {e}")
            return "", 0.0


# ============================================================================
# Claude Vision Analyzer (Fallback)
# ============================================================================

class ClaudeVisionAnalyzer:
    """Claude Vision for product identification and style classification."""

    def __init__(self, product_list: list, label_text_map: dict = None,
                 vision_prompt: str = None):
        self.client = anthropic.Anthropic()
        self.product_list = product_list
        self.label_text_map = label_text_map or {}
        self.vision_prompt = vision_prompt

    def analyze_image(self, image_bytes: bytes, filename: str,
                      mime_type: str) -> MatchResult:
        """Analyze image when OCR didn't find text."""
        if needs_processing(image_bytes):
            image_bytes, mime_type = resize_image_for_api(image_bytes, filename)

        prompt = self.vision_prompt or self._default_prompt()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )

            response_text = response.content[0].text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group())
                product_guess = result.get('product_guess', 'Unknown')
                confidence = float(result.get('confidence', 0.0))

                # Validate product name
                if product_guess not in self.product_list and product_guess != "Unknown":
                    for valid_product in self.product_list:
                        if product_guess.lower() in valid_product.lower():
                            product_guess = valid_product
                            break
                    else:
                        product_guess = "Unknown"
                        confidence = 0.40

                if product_guess == "Unknown" or product_guess.lower() == "unknown":
                    confidence = min(confidence, 0.50)

                return MatchResult(
                    product_title=product_guess,
                    confidence=confidence,
                    method=MatchMethod.VISION,
                    details=result.get('visible_clues', '')[:80]
                )

        except Exception as e:
            print(f"    Claude Vision error: {e}")

        return MatchResult(
            product_title="Unknown",
            confidence=0.0,
            method=MatchMethod.UNMATCHED,
            details="Vision analysis failed"
        )

    def classify_style(self, image_bytes: bytes, filename: str,
                       mime_type: str) -> str:
        """Classify photo style for sub-folder organization."""
        if needs_processing(image_bytes):
            image_bytes, mime_type = resize_image_for_api(image_bytes, filename)

        prompt = """Classify this product photo into ONE category:

1. "Product Shot" - Product is clearly the main focus, clean/simple background
2. "White Background" - Studio shot with pure white or very light background
3. "Lifestyle" - Product shown in context (counter, desk, routine setup)
4. "Model" - A person is visible in the image

RESPOND WITH ONLY ONE OF: Product Shot, White Background, Lifestyle, Model"""

        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )

            style = response.content[0].text.strip().lower()
            if "white" in style:
                return "White Background"
            elif "lifestyle" in style:
                return "Lifestyle"
            elif "model" in style:
                return "Model"
            else:
                return "Product Shot"

        except Exception as e:
            print(f"    Style classification error: {e}")
            return "Product Shot"

    def _default_prompt(self):
        products_str = "\n".join(f"- {p}" for p in self.product_list)
        return f"""Identify the product in this image from this list:

{products_str}

RESPOND WITH JSON:
{{
    "product_guess": "Product name from list above, or Unknown",
    "confidence": 0.0 to 1.0,
    "visible_clues": "Describe what you see that led to your guess"
}}

If you cannot confidently identify the product, respond with "Unknown"."""


# ============================================================================
# Product Catalog
# ============================================================================

class ProductCatalog:
    """Product catalog with keyword matching."""

    def __init__(self, catalog_path: str = "products.json"):
        self.products = []
        self.label_text_map = {}

        if os.path.exists(catalog_path):
            with open(catalog_path, 'r') as f:
                data = json.load(f)
            self.products = data.get('products', [])
            self.label_text_map = data.get('label_text_map', {})
        else:
            print(f"  [!] No catalog found at {catalog_path}")
            print("      Create one from products_example.json")

        self._build_keyword_index()

    def _build_keyword_index(self):
        self.keyword_map = {}
        for product in self.products:
            for keyword in product.get('keywords', []):
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_map:
                    self.keyword_map[keyword_lower] = []
                self.keyword_map[keyword_lower].append(product['title'])

    def match_filename(self, filename: str) -> Optional[MatchResult]:
        """Match product from filename."""
        filename_lower = filename.lower()
        name_clean = Path(filename_lower).stem
        name_clean = re.sub(r'[_\-\.]', ' ', name_clean)

        best_match = None
        best_score = 0

        for product in self.products:
            score = 0
            for keyword in product.get('keywords', []):
                if keyword.lower() in name_clean:
                    score += len(keyword.split())

            if score > best_score:
                best_score = score
                best_match = product

        if best_match and best_score > 0:
            confidence = min(0.6 + (best_score * 0.1), 0.95)
            return MatchResult(
                product_title=best_match['title'],
                confidence=confidence,
                method=MatchMethod.FILENAME,
                details="Matched keywords in filename"
            )

        return None

    def match_ocr_text(self, ocr_text: str) -> Tuple[str, float]:
        """Match extracted OCR text to a product."""
        if not ocr_text:
            return "Unknown", 0.0

        text_upper = ocr_text.upper()
        text_clean = re.sub(r'[^A-Z0-9\s-]', '', text_upper)

        # Try exact matches from label_text_map
        for label_text, product_name in self.label_text_map.items():
            if label_text.upper() in text_upper:
                return product_name, 0.95

        # Try cleaned text
        for label_text, product_name in self.label_text_map.items():
            label_clean = re.sub(r'[^A-Z0-9\s-]', '', label_text.upper())
            if label_clean in text_clean:
                return product_name, 0.90

        # Fuzzy match via keyword overlap
        best_match = None
        best_score = 0

        for label_text, product_name in self.label_text_map.items():
            label_words = set(label_text.upper().split())
            text_words = set(text_clean.split())
            overlap = label_words.intersection(text_words)
            if overlap:
                score = len(overlap) / len(label_words)
                if score > best_score:
                    best_score = score
                    best_match = product_name

        if best_match and best_score >= 0.5:
            return best_match, 0.70 + (best_score * 0.2)

        return "Unknown", 0.0

    def get_product_list(self) -> list:
        return [p['title'] for p in self.products]


# ============================================================================
# Google Drive Manager
# ============================================================================

class GoogleDriveManager:
    """Google Drive operations."""

    def __init__(self, credentials_path="credentials.json", token_path="token.json"):
        self.creds = self._get_credentials(credentials_path, token_path)
        self.service = build('drive', 'v3', credentials=self.creds)
        self._folder_cache = {}

    def _get_credentials(self, credentials_path, token_path) -> Credentials:
        creds = None

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(
                        f"Missing {credentials_path}. "
                        "Download OAuth credentials from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(token_path, 'w') as token:
                token.write(creds.to_json())

        return creds

    def list_images(self, folder_id: str) -> list:
        query = f"'{folder_id}' in parents and trashed=false"
        all_files = []
        page_token = None

        while True:
            results = self.service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token,
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()

            for f in results.get('files', []):
                ext = Path(f['name']).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS or f['mimeType'].startswith('image/'):
                    all_files.append(f)

            page_token = results.get('nextPageToken')
            if not page_token:
                break

        return all_files

    def download_file(self, file_id: str) -> bytes:
        request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buffer.seek(0)
        return buffer.getvalue()

    def create_folder(self, name: str, parent_id: str) -> str:
        cache_key = f"{parent_id}/{name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        query = (f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
                 f"and '{parent_id}' in parents and trashed=false")
        results = self.service.files().list(
            q=query, spaces='drive', fields='files(id)',
            supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()

        files = results.get('files', [])
        if files:
            self._folder_cache[cache_key] = files[0]['id']
            return files[0]['id']

        metadata = {
            'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        folder = self.service.files().create(
            body=metadata, fields='id', supportsAllDrives=True
        ).execute()

        self._folder_cache[cache_key] = folder['id']
        return folder['id']

    def move_file(self, file_id: str, new_folder_id: str):
        file = self.service.files().get(
            fileId=file_id, fields='parents', supportsAllDrives=True
        ).execute()
        previous_parents = ",".join(file.get('parents', []))

        self.service.files().update(
            fileId=file_id,
            addParents=new_folder_id,
            removeParents=previous_parents,
            fields='id',
            supportsAllDrives=True
        ).execute()


# ============================================================================
# Main Organizer
# ============================================================================

class PhotoOrganizer:
    """Main photo organization orchestrator."""

    def __init__(self, catalog_path: str = "products.json",
                 credentials_path: str = "credentials.json",
                 vision_prompt: str = None):
        print("=" * 60)
        print("Google Drive Photo Organizer")
        print("Hybrid OCR + Vision Pipeline")
        print("=" * 60)

        print("\nInitializing components...")
        self.catalog = ProductCatalog(catalog_path)
        self.drive = GoogleDriveManager(credentials_path)
        print("  [+] Google Drive connected")

        self.ocr = CloudVisionOCR()
        self.vision = ClaudeVisionAnalyzer(
            self.catalog.get_product_list(),
            self.catalog.label_text_map,
            vision_prompt
        )
        print("  [+] Claude Vision ready")

        self.stats = defaultdict(int)

    def organize(self, source_folder_id: str, create_organized_parent: bool = True,
                 batch_size: int = None, start_batch: int = 1,
                 classify_styles: bool = False):
        """Organize photos into product folders."""
        print(f"\nScanning folder: {source_folder_id}")
        all_images = self.drive.list_images(source_folder_id)
        total_images = len(all_images)
        print(f"Found {total_images} images\n")

        if total_images == 0:
            print("No images found!")
            return

        if create_organized_parent:
            output_folder_id = self.drive.create_folder("Organized Photos", source_folder_id)
        else:
            output_folder_id = source_folder_id

        review_folder_id = self.drive.create_folder("_Review", output_folder_id)
        multi_folder_id = self.drive.create_folder("_Multiple Products", output_folder_id)

        images = all_images
        if batch_size:
            total_batches = (total_images + batch_size - 1) // batch_size
            start_idx = (start_batch - 1) * batch_size
            end_idx = min(start_idx + batch_size, total_images)

            if start_idx >= total_images:
                print(f"Batch {start_batch} exceeds total. Max batch: {total_batches}")
                return

            images = all_images[start_idx:end_idx]
            print(f"Processing batch {start_batch}/{total_batches}")
            print(f"Images {start_idx + 1} to {end_idx} of {total_images}\n")

        self.stats['total'] = len(images)
        images = sorted(images, key=lambda x: x['name'])

        for i, image in enumerate(images, 1):
            filename = image['name']
            file_id = image['id']
            mime_type = image.get('mimeType', 'image/jpeg')

            print(f"\n[{i}/{len(images)}] {filename}")

            try:
                match = self._identify_product(file_id, filename, mime_type)

                if match.multiple_products:
                    print(f"  -> Multiple products ({match.confidence:.0%})")
                    self.stats['multi_product'] += 1
                    self.drive.move_file(file_id, multi_folder_id)

                elif match.confidence >= CONFIDENCE_THRESHOLD and match.product_title != "Unknown":
                    print(f"  -> {match.product_title} ({match.confidence:.0%}) [{match.method.value}]")
                    if match.details:
                        print(f"    {match.details[:80]}")

                    product_folder_id = self.drive.create_folder(match.product_title, output_folder_id)

                    if classify_styles and match.style:
                        style_folder_id = self.drive.create_folder(match.style, product_folder_id)
                        self.drive.move_file(file_id, style_folder_id)
                    else:
                        self.drive.move_file(file_id, product_folder_id)

                    self.stats[f'{match.method.value}_matched'] += 1

                else:
                    print(f"  ? Low confidence: {match.product_title} ({match.confidence:.0%})")
                    print(f"    -> Review folder")
                    self.stats['review'] += 1
                    self.drive.move_file(file_id, review_folder_id)

            except Exception as e:
                print(f"  [X] Error: {e}")
                self.stats['errors'] += 1

        self._print_summary(batch_size, start_batch, total_images)

    def _identify_product(self, file_id: str, filename: str, mime_type: str) -> MatchResult:
        """Hybrid pipeline: filename -> OCR -> Claude Vision."""
        # Step 1: Filename matching
        match = self.catalog.match_filename(filename)
        if match and match.confidence >= CONFIDENCE_THRESHOLD:
            return match

        # Step 2: Download for analysis
        print(f"  -> Downloading for analysis...")
        image_bytes = self.drive.download_file(file_id)

        # Step 3: Google Cloud Vision OCR
        print(f"  -> Running OCR...")
        ocr_text, ocr_confidence = self.ocr.extract_text(image_bytes)

        if ocr_text:
            display = ocr_text[:60] + "..." if len(ocr_text) > 60 else ocr_text
            print(f'    OCR found: "{display}"')

            product, text_confidence = self.catalog.match_ocr_text(ocr_text)

            if product != "Unknown" and text_confidence >= CONFIDENCE_THRESHOLD:
                return MatchResult(
                    product_title=product,
                    confidence=text_confidence,
                    method=MatchMethod.OCR,
                    details=f"Label text: '{ocr_text[:50]}'",
                    detected_text=ocr_text
                )
        else:
            print(f"    OCR: No text detected")

        # Step 4: Claude Vision fallback
        print(f"  -> Claude Vision fallback...")
        return self.vision.analyze_image(image_bytes, filename, mime_type)

    def _print_summary(self, batch_size, current_batch, total_images):
        print("\n" + "=" * 60)
        print("ORGANIZATION COMPLETE" if not batch_size else f"BATCH {current_batch} COMPLETE")
        print("=" * 60)
        print(f"Total processed:     {self.stats['total']}")
        print(f"Filename matches:    {self.stats.get('filename_matched', 0)}")
        print(f"OCR matches:         {self.stats.get('ocr_matched', 0)}")
        print(f"Vision matches:      {self.stats.get('vision_matched', 0)}")
        print(f"Multiple products:   {self.stats['multi_product']}")
        print(f"Sent to Review:      {self.stats['review']}")
        print(f"Errors:              {self.stats['errors']}")

        success = (self.stats.get('filename_matched', 0) +
                   self.stats.get('ocr_matched', 0) +
                   self.stats.get('vision_matched', 0))
        if self.stats['total'] > 0:
            print(f"\nSuccess rate: {success}/{self.stats['total']} ({100*success/self.stats['total']:.1f}%)")

        if batch_size and total_images:
            total_batches = (total_images + batch_size - 1) // batch_size
            remaining = total_images - (current_batch * batch_size)
            if remaining > 0:
                print(f"\nNext: python gdrive_photo_organizer.py FOLDER_ID --batch {batch_size} --start-batch {current_batch + 1}")

        print("=" * 60)


# ============================================================================
# Phase 2: Style Sub-Sorting
# ============================================================================

class StyleOrganizer:
    """Phase 2: Organize photos within product folders by style."""

    def __init__(self, credentials_path="credentials.json"):
        print("Style Organizer - Phase 2")
        self.drive = GoogleDriveManager(credentials_path)
        self.vision = ClaudeVisionAnalyzer([])
        self.stats = defaultdict(int)

    def organize_by_style(self, product_folder_id: str, batch_size: int = None):
        print(f"\nScanning product folder...")
        images = self.drive.list_images(product_folder_id)
        print(f"Found {len(images)} images\n")

        if not images:
            return

        style_folders = {}
        for style in STYLE_CATEGORIES:
            style_folders[style] = self.drive.create_folder(style, product_folder_id)

        for i, image in enumerate(images, 1):
            filename = image['name']
            file_id = image['id']
            mime_type = image.get('mimeType', 'image/jpeg')

            print(f"[{i}/{len(images)}] {filename}")

            try:
                image_bytes = self.drive.download_file(file_id)
                style = self.vision.classify_style(image_bytes, filename, mime_type)
                print(f"  -> {style}")
                self.drive.move_file(file_id, style_folders[style])
                self.stats[style] += 1
            except Exception as e:
                print(f"  [X] Error: {e}")
                self.stats['errors'] += 1

        print("\n" + "=" * 40)
        print("STYLE ORGANIZATION COMPLETE")
        print("=" * 40)
        for style in STYLE_CATEGORIES:
            print(f"{style}: {self.stats.get(style, 0)}")
        print(f"Errors: {self.stats['errors']}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AI-powered Google Drive photo organizer using OCR + Vision'
    )
    parser.add_argument('folder_id', help='Google Drive folder ID to organize')
    parser.add_argument('--catalog', default='products.json',
                        help='Product catalog JSON file (default: products.json)')
    parser.add_argument('--credentials', default='credentials.json',
                        help='Google OAuth credentials path (default: credentials.json)')
    parser.add_argument('--in-place', action='store_true',
                        help='Organize in place (no "Organized Photos" parent)')
    parser.add_argument('--confidence', type=float, default=0.60,
                        help='Confidence threshold (default: 0.60)')
    parser.add_argument('--batch', type=int,
                        help='Process in batches of this size')
    parser.add_argument('--start-batch', type=int, default=1,
                        help='Which batch to start from (default: 1)')
    parser.add_argument('--with-styles', action='store_true',
                        help='Also classify into style sub-folders')
    parser.add_argument('--style-only', action='store_true',
                        help='Phase 2: style classification only on existing folder')

    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence

    if args.style_only:
        organizer = StyleOrganizer(args.credentials)
        organizer.organize_by_style(args.folder_id, args.batch)
    else:
        organizer = PhotoOrganizer(
            catalog_path=args.catalog,
            credentials_path=args.credentials
        )
        organizer.organize(
            source_folder_id=args.folder_id,
            create_organized_parent=not args.in_place,
            batch_size=args.batch,
            start_batch=args.start_batch,
            classify_styles=args.with_styles
        )


if __name__ == '__main__':
    main()
