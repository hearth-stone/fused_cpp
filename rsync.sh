ARM_CODEX_HOST="${ARM_CODEX_HOST:-Arm-codex-internal}"

rsync -avzP \
    --copy-links \
    --exclude='.git/' \
    --exclude='.DS_Store' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.so' \
    --exclude='*.dylib' \
    --exclude='*.o' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='*.egg-info/' \
    --exclude='.eggs/' \
    --exclude='.hypothesis/' \
    --exclude='.pytest_cache/' \
    --exclude='.benchmarks/' \
    --exclude='.codebuddy/' \
    --exclude='.claude/' \
    --exclude='.gemini/' \
    --exclude='.venv/' \
    --exclude='bench/sdpa/sdpa_versions_*.csv' \
    --exclude='bench/sdpa/sdpa_versions_*.json' \
    --exclude='.codegraph/' \
    ./ "${ARM_CODEX_HOST}:/home/zhangxu/code"
