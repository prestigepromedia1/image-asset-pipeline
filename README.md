# Google Drive Photo Organizer

AI-powered photo organizer that automatically sorts product photos into folders using a hybrid OCR + Vision pipeline.

## How It Works

1. **Filename matching** - Fast keyword-based matching (free, no API calls)
2. **Google Cloud Vision OCR** - Reads text on product labels with high accuracy
3. **Claude Vision fallback** - AI visual analysis for photos where text isn't readable
4. **Style classification** (optional) - Sorts into sub-folders: Product Shot, White Background, Lifestyle, Model

Low-confidence matches go to a `_Review` folder for manual sorting.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Google Drive API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the **Google Drive API**
3. Create OAuth 2.0 credentials (Desktop app)
4. Download as `credentials.json` in this directory

### 3. Google Cloud Vision API

1. In the same Cloud project, enable **Cloud Vision API**
2. Set up authentication ([Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials) or service account)

### 4. Anthropic API Key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 5. Product Catalog

Copy the example and customize for your products:

```bash
cp products_example.json products.json
```

Edit `products.json` with your products:

```json
{
  "products": [
    {
      "title": "Your Product Name",
      "keywords": ["keyword1", "keyword2"]
    }
  ],
  "label_text_map": {
    "LABEL TEXT ON PRODUCT": "Your Product Name"
  }
}
```

### 6. Get Your Drive Folder ID

From the Google Drive URL: `https://drive.google.com/drive/folders/YOUR_FOLDER_ID`

## Usage

```bash
# Basic usage
python gdrive_photo_organizer.py YOUR_FOLDER_ID

# Organize in place (no parent folder created)
python gdrive_photo_organizer.py YOUR_FOLDER_ID --in-place

# Adjust confidence threshold
python gdrive_photo_organizer.py YOUR_FOLDER_ID --confidence 0.75

# Process in batches of 50
python gdrive_photo_organizer.py YOUR_FOLDER_ID --batch 50

# Continue from batch 3
python gdrive_photo_organizer.py YOUR_FOLDER_ID --batch 50 --start-batch 3

# Include style sub-folders (Phase 2)
python gdrive_photo_organizer.py YOUR_FOLDER_ID --with-styles

# Style-only pass on an already-organized product folder
python gdrive_photo_organizer.py PRODUCT_FOLDER_ID --style-only
```

## Output Structure

```
Source Folder/
  Organized Photos/
    Product Alpha/
      photo1.jpg
      photo2.jpg
    Product Beta Cream/
      cream_shot.png
    _Review/
      unclear_image.jpg
    _Multiple Products/
      group_shot.jpg
```

With `--with-styles`:

```
Product Alpha/
  Product Shot/
  White Background/
  Lifestyle/
  Model/
```

## Cost

- **Google Drive API**: Free within quotas
- **Cloud Vision OCR**: ~$1.50 per 1000 images
- **Claude Vision**: ~$0.01-0.03 per image (only used when OCR fails)
- **Filename matching**: Free (tried first)

## Troubleshooting

**"Missing credentials.json"** - Download from Google Cloud Console.

**"Token expired"** - Delete `token.json` and re-run.

**"Cloud Vision unavailable"** - Check that Vision API is enabled and credentials are set up.

**Low accuracy** - Improve your `products.json` label_text_map with more text variants.

## License

MIT
