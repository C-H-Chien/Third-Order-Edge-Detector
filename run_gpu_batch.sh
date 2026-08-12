#!/bin/bash
# =============================================================================
# Batch-run the GPU Third-Order Edge Detector (TOED) on a folder of images.
#
# Usage:
#   ./run_gpu_batch.sh <input_dir> <output_dir> [options]
#
# Arguments:
#   input_dir               Directory containing input images
#   output_dir              Directory where edge lists will be written
#                           (passed through to TOED as its 4th argument)
#
# Optional:
#   -t, --threads N         OpenMP CPU threads passed to TOED (default: 4)
#   -g, --gpu-id ID         CUDA device id (default: 0)
#   -e, --ext LIST          Comma-separated extensions (default: png,jpg,jpeg,pgm,bmp,tif,tiff)
#   --save-curvelets        Also keep chain.txt / info.txt when present
#   --load-modules          Source load_oscars_modules.sh before running
#   -h, --help              Show this help
#
# For each image, TOED writes data_final_output_gpu.txt into <output_dir>.
# This script renames that file to:
#   <output_dir>/<image_basename>_edges_gpu.txt
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOED_BIN="${SCRIPT_DIR}/TOED"
GPU_EDGE_FILE="data_final_output_gpu.txt"

INPUT_DIR=""
OUTPUT_DIR=""
NTHREADS=4
GPU_ID=0
EXTENSIONS="png,jpg,jpeg,JPEG,pgm,bmp,tif,tiff"
SAVE_CURVELETS=0
LOAD_MODULES=0

usage() {
    cat <<'EOF'
Batch-run the GPU Third-Order Edge Detector (TOED) on a folder of images.

Usage:
  ./run_gpu_batch.sh <input_dir> <output_dir> [options]

Arguments:
  input_dir               Directory containing input images
  output_dir              Directory where edge lists will be written
                          (passed through to TOED as its 4th argument)

Optional:
  -t, --threads N         OpenMP CPU threads passed to TOED (default: 4)
  -g, --gpu-id ID         CUDA device id (default: 0)
  -e, --ext LIST          Comma-separated extensions
                          (default: png,jpg,jpeg,pgm,bmp,tif,tiff)
  --save-curvelets        Also keep chain.txt / info.txt when present
  --load-modules          Source load_oscars_modules.sh before running
  -h, --help              Show this help

For each image, TOED writes data_final_output_gpu.txt into <output_dir>.
This script renames that file to:
  <output_dir>/<image_basename>_edges_gpu.txt

Example:
  ./run_gpu_batch.sh ./input_images ./my_edges -t 4 -g 0
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--threads)
            NTHREADS="${2:-}"; shift 2 ;;
        -g|--gpu-id)
            GPU_ID="${2:-}"; shift 2 ;;
        -e|--ext)
            EXTENSIONS="${2:-}"; shift 2 ;;
        --save-curvelets)
            SAVE_CURVELETS=1; shift ;;
        --load-modules)
            LOAD_MODULES=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        -*)
            die "Unknown option: $1 (use --help)" ;;
        *)
            if [[ -z "${INPUT_DIR}" ]]; then
                INPUT_DIR="$1"
            elif [[ -z "${OUTPUT_DIR}" ]]; then
                OUTPUT_DIR="$1"
            else
                die "Unexpected argument: $1 (use --help)"
            fi
            shift
            ;;
    esac
done

[[ -n "${INPUT_DIR}" ]]  || die "Missing input_dir (use --help)"
[[ -n "${OUTPUT_DIR}" ]] || die "Missing output_dir (use --help)"
[[ -d "${INPUT_DIR}" ]]  || die "Input directory not found: ${INPUT_DIR}"
[[ -x "${TOED_BIN}" ]]   || die "TOED binary not found or not executable: ${TOED_BIN}
Build with: make -f makefile.gpu_cpu"

if [[ "${LOAD_MODULES}" -eq 1 ]]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/load_oscars_modules.sh"
fi

mkdir -p "${OUTPUT_DIR}"

# Build a null-delimited find expression for the requested extensions
IFS=',' read -r -a EXT_ARR <<< "${EXTENSIONS}"
FIND_ARGS=()
for i in "${!EXT_ARR[@]}"; do
    ext="${EXT_ARR[$i]// /}"
    [[ -n "${ext}" ]] || continue
    if [[ ${#FIND_ARGS[@]} -gt 0 ]]; then
        FIND_ARGS+=( -o )
    fi
    FIND_ARGS+=( -iname "*.${ext}" )
done
[[ ${#FIND_ARGS[@]} -gt 0 ]] || die "No valid extensions in: ${EXTENSIONS}"

mapfile -d '' IMAGES < <(find "${INPUT_DIR}" -maxdepth 1 -type f \( "${FIND_ARGS[@]}" \) -print0 | sort -z)

NUM_IMAGES=${#IMAGES[@]}
[[ "${NUM_IMAGES}" -gt 0 ]] || die "No images found in ${INPUT_DIR} with extensions: ${EXTENSIONS}"

echo "=============================================="
echo " GPU TOED batch"
echo "=============================================="
echo " Input dir : ${INPUT_DIR}"
echo " Output dir: ${OUTPUT_DIR}"
echo " Images    : ${NUM_IMAGES}"
echo " Threads   : ${NTHREADS}"
echo " GPU id    : ${GPU_ID}"
echo " Binary    : ${TOED_BIN}"
echo "=============================================="

FAILED=0
SUCCEEDED=0

# Run from project root so relative defaults still resolve if needed
cd "${SCRIPT_DIR}"

for img in "${IMAGES[@]}"; do
    # mapfile keeps a trailing empty element when using -d ''
    [[ -n "${img}" ]] || continue

    base="$(basename "${img}")"
    stem="${base%.*}"
    out_edges="${OUTPUT_DIR}/${stem}_edges_gpu.txt"

    echo
    echo "[$((SUCCEEDED + FAILED + 1))/${NUM_IMAGES}] Processing: ${base}"

    if ! "${TOED_BIN}" "${img}" "${NTHREADS}" "${GPU_ID}" "${OUTPUT_DIR}"; then
        echo "  FAILED: TOED returned non-zero for ${base}" >&2
        FAILED=$((FAILED + 1))
        continue
    fi

    staged="${OUTPUT_DIR}/${GPU_EDGE_FILE}"
    if [[ ! -f "${staged}" ]]; then
        echo "  FAILED: expected output missing: ${staged}" >&2
        FAILED=$((FAILED + 1))
        continue
    fi

    mv -f "${staged}" "${out_edges}"
    echo "  Wrote edges -> ${out_edges}"

    if [[ "${SAVE_CURVELETS}" -eq 1 ]]; then
        for aux in chain.txt info.txt; do
            if [[ -f "${OUTPUT_DIR}/${aux}" ]]; then
                mv -f "${OUTPUT_DIR}/${aux}" "${OUTPUT_DIR}/${stem}_${aux}"
                echo "  Wrote ${aux}  -> ${OUTPUT_DIR}/${stem}_${aux}"
            fi
        done
    else
        # Avoid leaving shared intermediate names that the next image would overwrite
        rm -f "${OUTPUT_DIR}/chain.txt" "${OUTPUT_DIR}/info.txt" \
              "${OUTPUT_DIR}/data_final_output_cpu.txt"
    fi

    SUCCEEDED=$((SUCCEEDED + 1))
done

echo
echo "=============================================="
echo " Done. Succeeded: ${SUCCEEDED}  Failed: ${FAILED}"
echo "=============================================="

[[ "${FAILED}" -eq 0 ]]
