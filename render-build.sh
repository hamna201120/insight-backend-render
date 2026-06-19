#!/usr/bin/env bash
set -o errexit

echo "🚀 Installing build tools..."
pip install --upgrade pip setuptools wheel

echo "📦 Installing requirements..."
pip install -r requirements.txt --no-cache-dir

echo "🎤 Pre-downloading Whisper model..."
python -c "import whisper; whisper.load_model('tiny')"

echo "✅ Build complete!"
