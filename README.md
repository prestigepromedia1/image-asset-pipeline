# Image Asset Pipeline

AI-powered image categorization, metadata renaming, and indexing. Turn a folder of `IMG_4530.jpg` files into organized, properly named, searchable assets.

## The Problem

Brands, agencies, and studios have thousands of product images with meaningless filenames like `DSC_0042.jpg` or `IMG_4530.jpg`. These are invisible to DAMs, search, SEO, and AI tools. Manual sorting takes hours.

## The Pipeline

```
organize  ->  rename  ->  index
```

| Step | What it does | Example |
|------|-------------|---------|
| **organize** | AI sorts images into category folders | `IMG_4530.jpg` moves to `NMF Hydrator/` |
| **rename** | Bulk rename with metadata-rich names | Becomes `NMF-Hydrator-Barrier-Repair_product-shot_001.jpg` |
| **index** | Export CSV/JSON asset manifest | Spreadsheet with every file, category, path, link |
| **dedup** | Find duplicate images by checksum | Reports exact dupes + same-name variants |

Works with **local folders** or **Google Drive**.

## Quick Start

```bash
pip install anthropic Pillow
export ANTHROPIC_API_KEY="your-key"
```

### Organize a local folder

```bash
python image_asset_pipeline.py organize ./product-photos --local --catalog categories.json
```

### Rename organized files with metadata

```bash
python image_asset_pipeline.py rename ./product-photos/Organized --local
```

### Export asset index

```bash
python image_asset_pipeline.py index ./product-photos/Organized --local --format csv
```

### Find duplicates

```bash
python image_asset_pipeline.py dedup ./product-photos --local
```

## Google Drive Mode

Same commands, without `--local`:

```bash
python image_asset_pipeline.py organize DRIVE_FOLDER_ID --catalog categories.json
python image_asset_pipeline.py rename DRIVE_FOLDER_ID
python image_asset_pipeline.py index DRIVE_FOLDER_ID --format both
python image_asset_pipeline.py dedup DRIVE_FOLDER_ID
```

Requires Google Drive API credentials (see [Drive Setup](#google-drive-setup) below).

## How Organize Works

Three-tier identification pipeline:

1. **Filename matching** -- Free, instant. Checks filenames against your keyword catalog.
2. **Google Cloud Vision OCR** -- Reads text on product labels/packaging. Optional but highly accurate.
3. **Claude Vision fallback** -- AI visual analysis for images where text isn't readable.

Low-confidence matches go to `_Review/` for manual sorting.

## Category Catalog

Create `categories.json` (see `categories_example.json`):

```json
{
  "categories": [
    {
      "title": "Your Product Name",
      "keywords": ["keyword1", "keyword2"]
    }
  ],
  "label_text_map": {
    "LABEL TEXT ON PACKAGING": "Your Product Name"
  },
  "style_categories": ["Product Shot", "White Background", "Lifestyle", "Model"]
}
```

The `label_text_map` maps text that OCR reads on product labels to your canonical category names.

## Naming Templates

Control how `rename` names your files:

```bash
python image_asset_pipeline.py rename ./Organized --local --template "{category}_{style}_{seq}"
```

| Token | Description | Example |
|-------|-------------|---------|
| `{category}` | Category/product name (slugified) | `NMF-Hydrator-Barrier-Repair` |
| `{style}` | Style sub-folder (if any) | `product-shot` |
| `{seq}` | Sequence number (001, 002...) | `001` |
| `{date}` | Today's date | `20260303` |

Default: `{category}_{style}_{seq}`

Use `--dry-run` to preview renames before executing.

## Command Reference

### organize

```bash
python image_asset_pipeline.py organize SOURCE [options]
  --local              Source is a local directory
  --catalog FILE       Category catalog JSON (default: categories.json)
  --credentials FILE   Google OAuth credentials (default: credentials.json)
  --in-place           Don't create "Organized" parent folder
  --confidence N       Match threshold (default: 0.60)
  --batch N            Process N images at a time
  --start-batch N      Resume from batch N
  --dry-run            Preview without moving files
```

### rename

```bash
python image_asset_pipeline.py rename SOURCE [options]
  --local              Source is local
  --template TPL       Naming template (default: {category}_{style}_{seq})
  --dry-run            Preview without renaming
```

### index

```bash
python image_asset_pipeline.py index SOURCE [options]
  --local              Source is local
  --format FMT         csv, json, or both (default: csv)
  --output FILE        Output file path
  --checksums          Include MD5 checksums (slower)
```

### dedup

```bash
python image_asset_pipeline.py dedup SOURCE [options]
  --local              Source is local
```

## Google Drive Setup

Only needed if you're not using `--local`.

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **Google Drive API**
3. Create OAuth 2.0 credentials (Desktop app)
4. Download as `credentials.json`
5. Install: `pip install google-auth google-auth-oauthlib google-api-python-client`

Optional for OCR: Enable **Cloud Vision API** and install `google-cloud-vision`.

## Cost

| Component | Cost |
|-----------|------|
| Filename matching | Free |
| Cloud Vision OCR | ~$1.50 per 1000 images |
| Claude Vision | ~$0.01-0.03 per image (fallback only) |
| Google Drive API | Free within quotas |

Filename matching is tried first. OCR handles most labeled products. Claude Vision is only used when OCR fails.

## License

MIT
