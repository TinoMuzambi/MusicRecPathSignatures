#!/bin/bash
# Complete Music Recommendation Pipeline
# Follows the workflow defined in code/README.md
# Usage: ./run_complete_pipeline.sh [--skip-step N] [--start-from-step N] [--skip-optional]

set -e  # Exit on error
set -u  # Exit on undefined variable

# Default configuration (can be overridden via environment variables)
TRACKS_CSV="${TRACKS_CSV:-./data/fma_metadata/tracks.csv}"
N_TRACKS="${N_TRACKS:-4000}"
SEED="${SEED:-2025}"
AUDIO_ROOT="${AUDIO_ROOT:-./data/fma_medium}"
OUTPUT_DIR="${OUTPUT_DIR:-./data/processed_tracks}"
SIMILARITY_MATRIX="${SIMILARITY_MATRIX:-./data/similarity_matrix.npz}"
N_JOBS="${N_JOBS:-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SIGNATURE_ORDER="${SIGNATURE_ORDER:-1}"
SKIP_OPTIONAL="${SKIP_OPTIONAL:-false}"

# Parse command-line arguments
START_FROM_STEP=1
SKIP_STEPS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-from-step)
            START_FROM_STEP="$2"
            shift 2
            ;;
        --skip-step)
            SKIP_STEPS+=("$2")
            shift 2
            ;;
        --skip-optional)
            SKIP_OPTIONAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--start-from-step N] [--skip-step N] [--skip-optional]"
            exit 1
            ;;
    esac
done

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function to check if step should be skipped
should_skip_step() {
    local step=$1
    if [[ $step -lt $START_FROM_STEP ]]; then
        return 0  # Skip (before start point)
    fi
    for skip in "${SKIP_STEPS[@]}"; do
        if [[ $step -eq $skip ]]; then
            return 0  # Skip
        fi
    done
    return 1  # Don't skip
}

# Helper function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${YELLOW}Warning: $1 not found. Skipping dependent steps.${NC}"
        return 1
    fi
    return 0
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Complete Music Recommendation Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Data Preparation and Processing
if ! should_skip_step 1; then
    echo -e "${GREEN}[Step 1/13] Processing tracks and extracting features...${NC}"
    python src/scripts/robust_track_processing.py \
        --tracks-csv "$TRACKS_CSV" \
        --n-tracks "$N_TRACKS" \
        --seed "$SEED" \
        --audio-root "$AUDIO_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --results-dir results/track_processing \
        ${N_JOBS:+--n-jobs "$N_JOBS"} \
        --log-level "$LOG_LEVEL"
    
    # Verify outputs
    if [ ! -f "$OUTPUT_DIR/selected_tracks.json" ] || [ ! -f "$OUTPUT_DIR/features.json" ]; then
        echo "Error: Step 1 failed - required output files not found."
        exit 1
    fi
    echo -e "${GREEN}✓ Step 1 completed${NC}\n"
else
    echo -e "${YELLOW}[Step 1/13] Skipped${NC}\n"
fi

# Step 2: Exploratory Data Analysis (EDA)
if ! should_skip_step 2; then
    echo -e "${GREEN}[Step 2/13] Generating exploratory data analysis...${NC}"
    
    if ! check_file "$OUTPUT_DIR/features.json" || ! check_file "$OUTPUT_DIR/selected_tracks.json"; then
        echo "Error: Step 2 requires outputs from Step 1. Exiting."
        exit 1
    fi
    
    python src/scripts/generate_eda_analysis.py \
        --tracks-json "$OUTPUT_DIR/selected_tracks.json" \
        --features-json "$OUTPUT_DIR/features.json" \
        --output-dir results/eda \
        --dpi 300 \
        --log-level "$LOG_LEVEL"
    
    echo -e "${GREEN}✓ Step 2 completed${NC}\n"
else
    echo -e "${YELLOW}[Step 2/13] Skipped${NC}\n"
fi

# Step 3: Path Signature Computation
if ! should_skip_step 3; then
    echo -e "${GREEN}[Step 3/13] Computing path signatures and similarity matrix...${NC}"
    
    if ! check_file "$OUTPUT_DIR/features.json" || ! check_file "$OUTPUT_DIR/selected_tracks.json"; then
        echo "Error: Step 3 requires outputs from Step 1. Exiting."
        exit 1
    fi
    
    python src/scripts/compute_similarity.py \
        --features_file "$OUTPUT_DIR/features.json" \
        --tracks-json "$OUTPUT_DIR/selected_tracks.json" \
        --output "$SIMILARITY_MATRIX" \
        --signature-order "$SIGNATURE_ORDER" \
        --results-dir results/similarity_computation \
        --log-level "$LOG_LEVEL"
    
    if [ ! -f "$SIMILARITY_MATRIX" ]; then
        echo "Error: Step 3 failed - similarity matrix not generated."
        exit 1
    fi
    echo -e "${GREEN}✓ Step 3 completed${NC}\n"
else
    echo -e "${YELLOW}[Step 3/13] Skipped${NC}\n"
fi

# Step 4: Baseline Comparison
if ! should_skip_step 4; then
    echo -e "${GREEN}[Step 4/13] Running baseline comparison...${NC}"
    
    if ! check_file "$OUTPUT_DIR/features.json" || ! check_file "$OUTPUT_DIR/selected_tracks.json"; then
        echo "Error: Step 4 requires outputs from Step 1. Exiting."
        exit 1
    fi
    
    python src/scripts/run_baseline_comparison.py \
        --features-file "$OUTPUT_DIR/features.json" \
        --tracks-json "$OUTPUT_DIR/selected_tracks.json" \
        --output-dir results/baseline_comparison \
        --n-users 200 \
        --test-ratio 0.15 \
        --validation-ratio 0.15 \
        --log-level "$LOG_LEVEL"
    
    echo -e "${GREEN}✓ Step 4 completed${NC}\n"
else
    echo -e "${YELLOW}[Step 4/13] Skipped${NC}\n"
fi

# Step 5: Ablation Studies (Optional)
if [[ "$SKIP_OPTIONAL" == "false" ]] && ! should_skip_step 5; then
    echo -e "${GREEN}[Step 5/13] Running ablation studies...${NC}"
    
    if ! check_file "$OUTPUT_DIR/selected_tracks.json"; then
        echo "Warning: Skipping ablation studies - required files not found."
    else
        python src/scripts/run_ablation_studies.py \
            --tracks-json "$OUTPUT_DIR/selected_tracks.json" \
            --features-dir "$OUTPUT_DIR" \
            --output-dir results/ablation_studies \
            --signature-orders "1,2,3,4" \
            --temperatures "0.1,0.5,1.0,2.0,5.0" \
            --similarity-metrics "cosine,euclidean,manhattan" \
            --log-level "$LOG_LEVEL"
        
        echo -e "${GREEN}✓ Step 5 completed${NC}\n"
    fi
else
    echo -e "${YELLOW}[Step 5/13] Skipped (optional)${NC}\n"
fi

# Step 6: Evaluation Suite
if ! should_skip_step 6; then
    echo -e "${GREEN}[Step 6/13] Running evaluation suite...${NC}"
    
    if ! check_file "$SIMILARITY_MATRIX" || ! check_file "$OUTPUT_DIR/selected_tracks.json"; then
        echo "Error: Step 6 requires outputs from Steps 1-3. Exiting."
        exit 1
    fi
    
    python src/scripts/run_evaluation_suite.py \
        --similarity-matrix "$SIMILARITY_MATRIX" \
        --tracks-json "$OUTPUT_DIR/selected_tracks.json" \
        --output-dir results/evaluation \
        --k-values "1,5,10,20" \
        --cv-folds 5 \
        --test-ratio 0.15 \
        --validation-ratio 0.15 \
        --seed "$SEED" \
        --log-level "$LOG_LEVEL"
    
    echo -e "${GREEN}✓ Step 6 completed${NC}\n"
else
    echo -e "${YELLOW}[Step 6/13] Skipped${NC}\n"
fi

# Step 7: Generate Predictions (needed for visualisations)
if ! should_skip_step 7; then
    echo -e "${GREEN}[Step 7/13] Generating predictions for analysis...${NC}"
    
    if ! check_file "$OUTPUT_DIR/features.json" || ! check_file "$OUTPUT_DIR/selected_tracks.json"; then
        echo "Error: Step 7 requires outputs from Step 1. Exiting."
        exit 1
    fi
    
    python src/scripts/generate_predictions.py \
        --features-file "$OUTPUT_DIR/features.json" \
        --tracks-json "$OUTPUT_DIR/selected_tracks.json" \
        --output-dir "$OUTPUT_DIR/predictions.json" \
        --signature-order "$SIGNATURE_ORDER" \
        --results-dir results/predictions \
        --log-level "$LOG_LEVEL"
    
    echo -e "${GREEN}✓ Step 7 completed${NC}\n"
else
    echo -e "${YELLOW}[Step 7/13] Skipped${NC}\n"
fi

# Step 8: Generate Visualisations
if ! should_skip_step 8; then
    echo -e "${GREEN}[Step 8/13] Generating visualisations...${NC}"
    
    if ! check_file "$OUTPUT_DIR/selected_tracks.json" || ! check_file "$SIMILARITY_MATRIX" || \
       ! check_file "$OUTPUT_DIR/features.json" || ! check_file "$OUTPUT_DIR/predictions.json"; then
        echo "Warning: Some required files missing. Skipping visualisations."
    else
        python src/scripts/generate_visualisations.py \
            --songs-json "$OUTPUT_DIR/selected_tracks.json" \
            --similarity-matrix "$SIMILARITY_MATRIX" \
            --features-json "$OUTPUT_DIR/features.json" \
            --predictions-json "$OUTPUT_DIR/predictions.json" \
            --plots-dir results/visualisations \
            --log-level "$LOG_LEVEL"
        
        echo -e "${GREEN}✓ Step 8 completed${NC}\n"
    fi
else
    echo -e "${YELLOW}[Step 8/13] Skipped${NC}\n"
fi

# Step 9: Generate Dissertation Figures
if ! should_skip_step 9; then
    echo -e "${GREEN}[Step 9/13] Generating dissertation figures...${NC}"
    
    if ! check_file "results/baseline_comparison/baseline_comparison_results.json" || \
       ! check_file "results/baseline_comparison/statistical_comparison.json"; then
        echo "Warning: Baseline comparison results not found. Skipping dissertation figures."
    else
        python src/scripts/generate_dissertation_figures.py \
            --output-dir results/dissertation_figures \
            --baseline-results results/baseline_comparison/baseline_comparison_results.json \
            --stats-json results/baseline_comparison/statistical_comparison.json \
            --log-level "$LOG_LEVEL"
        
        echo -e "${GREEN}✓ Step 9 completed${NC}\n"
    fi
else
    echo -e "${YELLOW}[Step 9/13] Skipped${NC}\n"
fi

# Step 10: Sync Figures for LaTeX
if ! should_skip_step 10; then
    echo -e "${GREEN}[Step 10/13] Syncing figures for LaTeX...${NC}"
    
    python src/scripts/sync_figures.py \
        --code-root . \
        --figures-dir ../figures
    
    echo -e "${GREEN}✓ Step 10 completed${NC}\n"
else
    echo -e "${YELLOW}[Step 10/13] Skipped${NC}\n"
fi

# Step 11: Generate Synthetic Users (Optional)
if [[ "$SKIP_OPTIONAL" == "false" ]] && ! should_skip_step 11; then
    echo -e "${GREEN}[Step 11/13] Generating synthetic users...${NC}"
    
    if ! check_file "$OUTPUT_DIR/selected_tracks.json"; then
        echo "Warning: Skipping synthetic user generation - required files not found."
    else
        python src/scripts/generate_synthetic_users.py \
            --tracks-json "$OUTPUT_DIR/selected_tracks.json" \
            --output-dir results/synthetic_users \
            --n-users 200 \
            --log-level "$LOG_LEVEL"
        
        echo -e "${GREEN}✓ Step 11 completed${NC}\n"
    fi
else
    echo -e "${YELLOW}[Step 11/13] Skipped (optional)${NC}\n"
fi

# Step 12: Run Robustness Analysis (Optional)
if [[ "$SKIP_OPTIONAL" == "false" ]] && ! should_skip_step 12; then
    echo -e "${GREEN}[Step 12/13] Running robustness analysis...${NC}"
    
    if ! check_file "$SIMILARITY_MATRIX" || ! check_file "$OUTPUT_DIR/selected_tracks.json"; then
        echo "Warning: Skipping robustness analysis - required files not found."
    else
        python src/scripts/run_robustness_analysis.py \
            --output-dir results/robustness \
            --similarity-matrix "$SIMILARITY_MATRIX" \
            --tracks-json "$OUTPUT_DIR/selected_tracks.json" \
            --log-level "$LOG_LEVEL"
        
        echo -e "${GREEN}✓ Step 12 completed${NC}\n"
    fi
else
    echo -e "${YELLOW}[Step 12/13] Skipped (optional)${NC}\n"
fi

# Step 13: Create Dissertation Package
if ! should_skip_step 13; then
    echo -e "${GREEN}[Step 13/13] Creating dissertation package...${NC}"
    
    python src/scripts/create_dissertation_package.py \
        --results-root results \
        --output-dir results/dissertation_package \
        --log-level "$LOG_LEVEL"
    
    echo -e "${GREEN}✓ Step 13 completed${NC}\n"
else
    echo -e "${YELLOW}[Step 13/13] Skipped${NC}\n"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Key outputs:"
echo "  - Processed tracks: $OUTPUT_DIR/"
echo "  - Track processing logs: results/track_processing/"
echo "  - EDA analysis: results/eda/"
echo "  - Similarity matrix: $SIMILARITY_MATRIX"
echo "  - Similarity computation logs: results/similarity_computation/"
echo "  - Baseline comparison: results/baseline_comparison/"
echo "  - Evaluation results: results/evaluation/"
echo "  - Predictions logs: results/predictions/"
echo "  - Visualisations: results/visualisations/"
echo "  - Dissertation figures: results/dissertation_figures/"
echo "  - Dissertation package: results/dissertation_package/"

