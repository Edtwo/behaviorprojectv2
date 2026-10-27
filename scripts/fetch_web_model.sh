#!/bin/sh
# Download the speech model the web app serves from its own site (web/models/, ~76 MB).
# Kept out of git (see .gitignore); run this once before `vercel deploy web --prod`.
# Model: Whisper base.en (MIT license, OpenAI), ONNX export by onnx-community, 8-bit weights.
set -e
cd "$(dirname "$0")/../web"
M=models/onnx-community/whisper-base.en_timestamped
B=https://huggingface.co/onnx-community/whisper-base.en_timestamped/resolve/main
mkdir -p "$M/onnx"
for f in config.json generation_config.json preprocessor_config.json tokenizer.json tokenizer_config.json \
         onnx/encoder_model_quantized.onnx onnx/decoder_model_merged_quantized.onnx; do
  [ -s "$M/$f" ] || curl -fL -o "$M/$f" "$B/$f"
done
du -sh models
