#!/bin/sh
set -e

if [ "${AUTO_DOWNLOAD_MODELS:-0}" = "1" ]; then
  python -m app.download_models
fi

exec python -m app.main

