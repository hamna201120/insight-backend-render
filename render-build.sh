#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip
pip install setuptools==69.5.1 wheel==0.42.0

pip install -r requirements.txt --no-cache-dir

echo "Build complete"
