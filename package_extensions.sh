#!/bin/bash
# VTOP Captcha Solver - Extension Packager
# Produces an unzipped Chrome folder and a zipped Firefox extension in the root directory.

# Exit on error
set -e

# Get directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXT_DIR="$SCRIPT_DIR/extension"
ROOT_DIR="$SCRIPT_DIR"

VERSION="1.3.1"
CHROME_OUT="$ROOT_DIR/captop-chrome"
FIREFOX_OUT="$ROOT_DIR/captop-firefox-v$VERSION.zip"

# Clean up old outputs in root
echo "🧹 Cleaning up old outputs..."
rm -rf "$CHROME_OUT"
rm -f "$ROOT_DIR"/captop-firefox-v*.zip
rm -f "$ROOT_DIR"/captop-chrome-v*.zip
rm -rf "$ROOT_DIR"/dist 

# 1. Package Chrome (manifest.json is currently Chrome)
echo "🌐 Preparing Chrome extension folder..."
mkdir -p "$CHROME_OUT"
# Copy all but exclude firefox specific files and hidden files
rsync -av --exclude="firefox-manifest.json" --exclude="manifest.json.chrome" --exclude=".*" "$EXT_DIR/" "$CHROME_OUT/"
echo "✅ Chrome folder ready at $CHROME_OUT"

# 2. Package Firefox
echo "🦊 Packaging Firefox extension v$VERSION..."
cd "$EXT_DIR"
# Temporarily swap manifests
if [ -f "firefox-manifest.json" ]; then
    mv manifest.json manifest.json.chrome
    cp firefox-manifest.json manifest.json
    
    zip -r "$FIREFOX_OUT" . -x "manifest.json.chrome" "firefox-manifest.json" "*.zip" ".*"
    
    # Restore Chrome manifest
    mv manifest.json.chrome manifest.json
    echo "✅ Firefox packaging complete at $FIREFOX_OUT"
else
    echo "❌ Error: firefox-manifest.json not found!"
    exit 1
fi

cd "$ROOT_DIR"
echo "🎉 Done! Outputs are in the project root."
ls -d "$CHROME_OUT"
ls -lh "$FIREFOX_OUT"
