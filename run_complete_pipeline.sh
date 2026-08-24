#!/usr/bin/env bash
# Complete, fail-closed dissertation release pipeline.
set -Eeuo pipefail

usage() {
    echo "Usage: $0 --run-root DIR --tracks-csv FILE --audio-root DIR --repository-root DIR --n-jobs N" >&2
}

RUN_ROOT=""
TRACKS_CSV=""
AUDIO_ROOT=""
REPOSITORY_ROOT=""
N_JOBS=""

while (($#)); do
    case "$1" in
        --run-root)
            RUN_ROOT="${2:?missing value for --run-root}"
            shift 2
            ;;
        --tracks-csv)
            TRACKS_CSV="${2:?missing value for --tracks-csv}"
            shift 2
            ;;
        --audio-root)
            AUDIO_ROOT="${2:?missing value for --audio-root}"
            shift 2
            ;;
        --repository-root)
            REPOSITORY_ROOT="${2:?missing value for --repository-root}"
            shift 2
            ;;
        --n-jobs)
            N_JOBS="${2:?missing value for --n-jobs}"
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$RUN_ROOT" || -z "$TRACKS_CSV" || -z "$AUDIO_ROOT" || -z "$REPOSITORY_ROOT" || -z "$N_JOBS" ]]; then
    usage
    exit 2
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "Run root already exists: $RUN_ROOT" >&2
    exit 2
fi
if [[ ! -f "$TRACKS_CSV" || ! -d "$AUDIO_ROOT" || ! -d "$REPOSITORY_ROOT/.git" ]]; then
    echo "A required release input is absent" >&2
    exit 2
fi
if [[ "$N_JOBS" != "4" ]]; then
    echo "--n-jobs must equal the frozen release value 4" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0

ARGS=(
    --run-root "$RUN_ROOT"
    --tracks-csv "$TRACKS_CSV"
    --audio-root "$AUDIO_ROOT"
    --repository-root "$REPOSITORY_ROOT"
    --n-jobs "$N_JOBS"
)

timeout --signal=TERM --kill-after=60s 43140s \
    python -m src.scripts.run_release_pipeline "${ARGS[@]}"
