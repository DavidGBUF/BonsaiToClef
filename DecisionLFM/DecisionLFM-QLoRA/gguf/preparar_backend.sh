#!/usr/bin/env bash
# Linux/CUDA. O checkout externo e os binários ficam fora dos arquivos versionados.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
llama=${1:-"$root/backends/llama.cpp"}
build=${2:-"$llama/build-decisionlfm"}
revision=0ec4027a2a0183de87c007069adddc06e665f7de
if [ ! -d "$llama/.git" ]; then
    git clone --no-checkout https://github.com/ggml-org/llama.cpp.git "$llama"
    git -C "$llama" checkout "$revision"
fi
if [ "$(git -C "$llama" rev-parse HEAD)" != "$revision" ]; then
    echo "Use um checkout llama.cpp no commit $revision" >&2
    exit 1
fi
patch=$root/gguf/llama-embedding-only.patch
if git -C "$llama" apply --check "$patch" 2>/dev/null; then
    git -C "$llama" apply "$patch"
elif ! git -C "$llama" apply --reverse --check "$patch" 2>/dev/null; then
    echo "O patch LFM3 não corresponde ao checkout fornecido" >&2
    exit 1
fi
cmake -S "$llama" -B "$build" -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-native}" -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build "$build" -j "${JOBS:-4}" --target llama llama-quantize
build=$(cd "$build" && pwd)
c++ -std=c++17 -O3 -shared -fPIC "$root/gguf/bridge.cpp" \
    -I"$llama/include" -I"$llama/src" -I"$llama/ggml/include" \
    -L"$build/bin" -Wl,-rpath,"$build/bin" -lllama -lggml -lggml-base -lggml-cpu \
    -o "$root/gguf/libdecisionlfm.so"
echo "Biblioteca criada: $root/gguf/libdecisionlfm.so"

