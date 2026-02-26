#!/bin/bash
# VTOP Captcha Solver - Extension Packager
# Produces a Chrome .zip and a Firefox .zip

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXT_DIR="$SCRIPT_DIR/extension"
ROOT_DIR="$SCRIPT_DIR"

# Version from argument or manifest
V_ARG="${1#v}"
VERSION="${V_ARG:-$(grep '"version"' "$EXT_DIR/manifest.json" | head -1 | sed 's/.*: *"\(.*\)".*/\1/')}"

CHROME_ZIP="$ROOT_DIR/captop-chrome-v$VERSION.zip"
FIREFOX_ZIP="$ROOT_DIR/captop-firefox-v$VERSION.zip"

# Clean up
echo "🧹 Cleaning up old outputs..."
rm -f "$ROOT_DIR"/captop-chrome-v*.zip
rm -f "$ROOT_DIR"/captop-firefox-v*.zip

# 1. Chrome
echo "🌐 Packaging Chrome extension v$VERSION..."
cd "$EXT_DIR"
zip -r "$CHROME_ZIP" . -x "firefox-manifest.json" "manifest.json.chrome" ".*" > /dev/null
echo "✅ Chrome .zip ready at $CHROME_ZIP"

# 2. Firefox
echo "🦊 Packaging Firefox extension v$VERSION..."
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
