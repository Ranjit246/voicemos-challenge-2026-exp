#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMO_SCRIPT="/Users/ranjitpatro/Home/Research/VoiceMOS/voicemos-challenge-2026-exp/NeMo/examples/speaker_tasks/recognition/extract_speaker_embeddings.py"
MANIFEST="$SCRIPT_DIR/manifest.json"
EMBED_DIR="$SCRIPT_DIR/embeddings"

# Step 1: generate manifest from train.csv + dev.csv
python3 "$SCRIPT_DIR/gen_manifest.py"

# Step 2: extract embeddings (saved to $EMBED_DIR/embeddings/manifest_embeddings.pt)
python3 "$NEMO_SCRIPT" \
    --manifest="$MANIFEST" \
    --model_path=titanet_large \
    --batch_size=1 \
    --embedding_dir="$EMBED_DIR"

echo "Done. Embeddings saved to $EMBED_DIR/embeddings/"
