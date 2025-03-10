#!/bin/bash

# Configuration: Model file path and expected SHA-256 checksum
MODEL_PATH="polaris/model/sft_loop.pt"
EXPECTED_HASH="cae9e9a28e5c3ff0d328934c066d275371d5301db084a914431198134f66ada2"

# Pre-check: Verify if the model file exists with valid checksum
if [ -f "$MODEL_PATH" ]; then
    # Calculate current file hash
    ACTUAL_HASH=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
    
    # Hash validation logic
    if [ "$ACTUAL_HASH" = "$EXPECTED_HASH" ]; then
        echo "✅ Valid model file detected, skipping download"
        pip install --use-pep517 --editable .
        echo "✅ Polaris installation completed"
        exit 0
    else
        # Security measure: Remove corrupted/invalid file
        echo "⚠️ Invalid file hash detected, triggering re-download"
        rm -f "$MODEL_PATH"
    fi
fi

# Model download process
echo "⏳ Downloading model from Hugging Face..."
wget -O "$MODEL_PATH" "https://huggingface.co/rr-ss/Polaris/resolve/main/polaris/model/sft_loop.pt?download=true"

# Post-download verification
ACTUAL_HASH=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
    # Error handling for failed verification
    rm -f "$MODEL_PATH"
    echo "❌ Download failed: Checksum mismatch (Actual: $ACTUAL_HASH)"
    echo "Manual download required:"
    echo "wget -O polaris/model/sft_loop.pt \"https://huggingface.co/rr-ss/Polaris/resolve/main/polaris/model/sft_loop.pt?download=true\""
    exit 1
else
    # Success workflow
    pip install --use-pep517 --editable .
    echo "✅ Model saved to: $MODEL_PATH"
    echo "✅ Polaris installed successfully"
fi