#!/usr/bin/env bash
# Run from any directory. Optional first argument overrides Python's include dir.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
CXX=${CXX:-g++}
SWIG=${SWIG:-swig}
PY_INCLUDE=${1:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')}
OMP_FLAGS=(-fopenmp)
LINK_FLAGS=(-shared)
if [[ $(uname -s) == Darwin ]]; then
    # Apple clang needs a separate OpenMP runtime (e.g. brew install libomp).
    : "${OPENMP_PREFIX:?Set OPENMP_PREFIX to the libomp installation prefix}"
    OMP_FLAGS=(-arch "$("$PYTHON" -c 'import platform; print(platform.machine())')" -Xpreprocessor -fopenmp "-I${OPENMP_PREFIX}/include")
    LINK_FLAGS+=(-undefined dynamic_lookup "-L${OPENMP_PREFIX}/lib" "-Wl,-rpath,${OPENMP_PREFIX}/lib" -lomp)
fi
"$SWIG" -c++ -python -outdir . cbinder/CTree.i
"$CXX" -std=c++17 -O3 -fPIC "${OMP_FLAGS[@]}" -I "$PY_INCLUDE" \
    cbinder/CTree_utils.cpp cbinder/CTree.cpp cbinder/frechet_consensus.cpp \
    cbinder/CTree_wrap.cxx "${LINK_FLAGS[@]}" -o _CTree.so
