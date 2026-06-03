#!/bin/bash
# ============================================
# CHIMBISIAI v1.1 — Quantization Script
# ============================================
# Конвертирует F16 GGUF в Q8_0 и Q4_K_M для продакшена
# Запускать на GPU-сервере ПОСЛЕ обучения и merge
#
# Требования: llama.cpp собран (make в /root/llama.cpp)
# Вход: F16 GGUF файл
# Выход: Q8_0 + Q4_K_M файлы
# ============================================

set -e

MODEL_DIR="/root/chimbisiai/output_v11"
F16_FILE="${MODEL_DIR}/chimbisiai-v11-f16.gguf"
Q8_FILE="${MODEL_DIR}/chimbisiai-v11-q8_0.gguf"
Q4_FILE="${MODEL_DIR}/chimbisiai-v11-q4_k_m.gguf"
LLAMA_CPP="/root/llama.cpp"

echo "=== CHIMBISIAI v1.1 Quantization ==="
echo "Input: ${F16_FILE}"
echo ""

# Step 1: Build llama.cpp if not exists
if [ ! -f "/root/llama.cpp/build/bin/llama-quantize" ]; then
    echo "[1/4] Building llama.cpp..."
    if [ ! -d "${LLAMA_CPP}" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git ${LLAMA_CPP}
    fi
    cd ${LLAMA_CPP}
    # Build with CUDA support for faster quantization
    make -j$(nproc) GGML_CUDA=1 2>/dev/null || make -j$(nproc)
    echo "  ✅ llama.cpp built"
else
    echo "[1/4] llama.cpp already built ✅"
fi

# Step 2: Check input file
if [ ! -f "${F16_FILE}" ]; then
    echo "❌ ERROR: F16 file not found: ${F16_FILE}"
    echo "Run training + merge + GGUF conversion first!"
    exit 1
fi
echo "[2/4] Input file: $(du -h ${F16_FILE} | cut -f1)"

# Step 3: Quantize to Q8_0 (best quality, ~8GB, 2x faster than F16)
echo "[3/4] Quantizing to Q8_0..."
/root/llama.cpp/build/bin/llama-quantize ${F16_FILE} ${Q8_FILE} q8_0
echo "  ✅ Q8_0: $(du -h ${Q8_FILE} | cut -f1)"

# Step 4: Quantize to Q4_K_M (production, ~4.5GB, 3-4x faster than F16)
echo "[4/4] Quantizing to Q4_K_M..."
/root/llama.cpp/build/bin/llama-quantize ${F16_FILE} ${Q4_FILE} q4_k_m
echo "  ✅ Q4_K_M: $(du -h ${Q4_FILE} | cut -f1)"

echo ""
echo "=== QUANTIZATION COMPLETE ==="
echo "Files:"
echo "  F16:    $(du -h ${F16_FILE} | cut -f1) — training/reference"
echo "  Q8_0:   $(du -h ${Q8_FILE} | cut -f1) — high quality production"
echo "  Q4_K_M: $(du -h ${Q4_FILE} | cut -f1) — fast production (recommended)"
echo ""
echo "Next steps:"
echo "  1. Deploy Q4_K_M to VPS-3 Ollama for production (fast, 5-10s responses)"
echo "  2. Keep Q8_0 as fallback if quality drops"
echo "  3. Keep F16 as archive"
echo ""
echo "Deploy command:"
echo "  scp ${Q4_FILE} root@<VPS-3>:/root/"
echo "  ssh root@<VPS-3> 'ollama create chimbisiai:v1.1 -f /root/Modelfile_v11'"
