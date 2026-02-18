#!/bin/bash
# VTOP Captcha Solver - Extension Packager
# Produces a Chrome .crx (if .pem provided) + .zip fallback, and a Firefox .zip

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXT_DIR="$SCRIPT_DIR/extension"
ROOT_DIR="$SCRIPT_DIR"

# Version from argument or manifest
VERSION="${1:-$(grep '"version"' "$EXT_DIR/manifest.json" | head -1 | sed 's/.*: *"\(.*\)".*/\1/')}"
PEM_FILE="${2:-$ROOT_DIR/captop-chrome.pem}"

CHROME_DIR="$ROOT_DIR/captop-chrome"
CHROME_CRX="$ROOT_DIR/captop-chrome.crx"
CHROME_ZIP="$ROOT_DIR/captop-chrome-v$VERSION.zip"
FIREFOX_ZIP="$ROOT_DIR/captop-firefox-v$VERSION.zip"

# Clean up
echo "🧹 Cleaning up old outputs..."
rm -rf "$CHROME_DIR" "$CHROME_CRX"
rm -f "$ROOT_DIR"/captop-chrome-v*.zip
rm -f "$ROOT_DIR"/captop-firefox-v*.zip

# 1. Chrome
echo "🌐 Packaging Chrome extension v$VERSION..."
mkdir -p "$CHROME_DIR"
rsync -a --exclude="firefox-manifest.json" --exclude="manifest.json.chrome" --exclude=".*" "$EXT_DIR/" "$CHROME_DIR/"

# Build .crx if pem exists, otherwise just zip
if [ -f "$PEM_FILE" ]; then
    # Create .crx using openssl
    TEMP_DIR=$(mktemp -d)
    cd "$CHROME_DIR"
    zip -r "$TEMP_DIR/extension.zip" . -x ".*" > /dev/null
    
    # Sign
    openssl sha1 -sign "$PEM_FILE" < "$TEMP_DIR/extension.zip" > "$TEMP_DIR/sig"
    openssl rsa -pubout -outform DER < "$PEM_FILE" > "$TEMP_DIR/pub" 2>/dev/null
    
    # Build CRX3-like package (simplified CRX format)
    {
        printf "Cr24"
        printf '\x02\x00\x00\x00'
        python3 -c "import struct,sys; pub=open('$TEMP_DIR/pub','rb').read(); sig=open('$TEMP_DIR/sig','rb').read(); sys.stdout.buffer.write(struct.pack('<II',len(pub),len(sig)))"
        cat "$TEMP_DIR/pub" "$TEMP_DIR/sig" "$TEMP_DIR/extension.zip"
    } > "$CHROME_CRX"
    
    rm -rf "$TEMP_DIR"
    cd "$ROOT_DIR"
    echo "✅ Chrome .crx ready at $CHROME_CRX"
fi

# Always produce a zip too
cd "$CHROME_DIR"
zip -r "$CHROME_ZIP" . -x ".*" > /dev/null
cd "$ROOT_DIR"
echo "✅ Chrome .zip ready at $CHROME_ZIP"

# Clean up temp folder
rm -rf "$CHROME_DIR"

# 2. Firefox
echo "🦊 Packaging Firefox extension v$VERSION..."
cd "$EXT_DIR"
if [ -f "firefox-manifest.json" ]; then
    mv manifest.json manifest.json.chrome
    cp firefox-manifest.json manifest.json

    zip -r "$FIREFOX_ZIP" . -x "manifest.json.chrome" "firefox-manifest.json" "*.zip" ".*" > /dev/null

    mv manifest.json.chrome manifest.json
    echo "✅ Firefox .zip ready at $FIREFOX_ZIP"
else
    echo "❌ Error: firefox-manifest.json not found!"
    exit 1
fi

cd "$ROOT_DIR"
echo "🎉 Done! Outputs:"
ls -lh "$CHROME_ZIP" "$FIREFOX_ZIP"
[ -f "$CHROME_CRX" ] && ls -lh "$CHROME_CRX"
