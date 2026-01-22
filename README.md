# Music Recommendation System with Path Signatures

A comprehensive music recommendation system that combines **novel path signature analysis** with **established recommendation methods** using LightFM and Implicit libraries. This MSc project demonstrates the effectiveness of path signatures for music recommendation through rigorous comparison with traditional approaches.

## Key Features

### **Novel Path Signature Method**

- **Path signature computation** for capturing musical structure using the esig library
- **Genre-based musical categories** using actual music metadata instead of arbitrary clustering
- **Optimised similarity computation** with focus on path signatures (80% weight)
- **Temperature scaling** for adjustable recommendation sensitivity
- **Default configuration**: Order 1 (best performance/latency trade-off), with ablations at orders 2–4

### **Traditional Baseline Methods**

- **Collaborative Filtering**: User-based CF using Implicit library, Item-based CF using LightFM
- **Content-based Filtering**: Traditional audio features with cosine similarity
- **Matrix Factorisation**: SVD, NMF, and Hybrid models using LightFM

### **Comprehensive Evaluation**

- **Standard Metrics**: Precision@K, Recall@K, F1@K, MAP@K
- **Advanced Metrics**: NDCG, Diversity, Novelty, Coverage, Serendipity
- **Performance Analysis**: Prediction time, computational efficiency
- **Baseline Comparison**: Rigorous evaluation against established methods

### **Audio Processing**

- Audio feature extraction (MFCCs, chroma, spectral features, zero crossing rate, tempo)
- Multi-dimensional time series creation combining pitch, loudness, tempo, and MFCCs
- **Robust track processing** with automatic retry mechanism for failed tracks
- **Memory-efficient processing** with configurable limits to prevent system crashes
- **Comprehensive error handling** for corrupted audio files and processing failures
- Parallel processing for improved performance
- Comprehensive logging system with configurable levels
- Command-line interface for easy usage
- **Visualisation tools** for similarity, category distribution, confusion matrix, and feature embeddings
- **Comprehensive evaluation framework** with recommendation metrics, classification metrics, and cross-validation

## Project Structure

```
code/
├── src/
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── processing.py
│   │   └── feature_extraction.py
│   ├── signatures/
│   │   ├── __init__.py
│   │   └── path_signatures.py
│   ├── analysis/
│   │   ├── similarity.py
│   │   ├── softmax_regression.py
│   │   ├── collaborative_filtering.py    # LightFM & Implicit
│   │   ├── content_based_filtering.py    # Traditional features
│   │   ├── matrix_factorization.py       # SVD, NMF, Hybrid
│   │   ├── evaluation_metrics.py         # Comprehensive metrics
│   │   └── visualisation.py
│   ├── recommendation/
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── recommendation_metrics.py
│   │   ├── classification_metrics.py
│   │   ├── cross_validation.py
│   │   └── README.md
│   └── utils/
│       ├── __init__.py
│       ├── logger_config.py
│       ├── signal_processing.py
│       └── metadata.py
├── src/scripts/
│   ├── robust_track_processing.py       # Robust track selection and processing
│   ├── audio_processing_main.py         # Audio processing and feature extraction
│   ├── compute_similarity.py             # Path signature computation
│   ├── generate_eda_analysis.py         # Comprehensive exploratory data analysis
│   ├── generate_synthetic_users.py      # Synthetic user generation for evaluation
│   ├── recommendation_main.py            # Path signature recommendations
│   ├── run_baseline_comparison.py        # Comprehensive baseline comparison
│   ├── run_ablation_studies.py           # Ablation studies framework
│   ├── run_evaluation_suite.py           # Complete evaluation framework
│   ├── run_robustness_analysis.py        # Robustness analysis (bootstrap, sensitivity)
│   ├── generate_visualisations.py        # Basic visualisations
│   ├── generate_dissertation_figures.py  # Publication-quality figures
│   ├── sync_figures.py                   # Sync figures for LaTeX
│   ├── generate_predictions.py           # Genre predictions
│   ├── find_cross_genre_similar_songs.py # Cross-genre similarity analysis
│   └── create_dissertation_package.py    # Package results for LaTeX
├── tests/
│   ├── test_baseline_methods.py
│   ├── test_similarity.py
│   ├── test_softmax_regression.py
│   ├── test_audio_processing.py
│   ├── test_feature_extraction.py
│   ├── test_metadata.py
│   ├── test_path_signatures.py
│   ├── test_recommendation.py
│   ├── test_visualisation.py
│   └── test_evaluation.py
├── data/                    # Requires FMA dataset (see Dataset Requirements section)
│   ├── fma_medium/         # Audio files (25,000 tracks)
│   └── fma_metadata/       # Track metadata
├── results/                 # All analysis outputs
│   ├── track_processing/   # Logs and timing from robust_track_processing.py
│   ├── eda/                # Exploratory data analysis
│   ├── similarity_computation/  # Logs and timing from compute_similarity.py
│   ├── baseline_comparison/# Baseline method comparison
│   ├── ablation_studies/   # Ablation study results
│   ├── evaluation/         # Evaluation suite results
│   ├── predictions/        # Logs and timing from generate_predictions.py
│   ├── visualisations/     # All generated plots
│   ├── dissertation_figures/  # Publication-quality figures
│   └── dissertation_package/  # Final package for LaTeX
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd code
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

If your environment does not support building optional libraries (e.g., LightFM/Implicit), the baselines automatically fall back to cosine-based implementations. Ensure `esig` installs correctly (it pulls binary deps). If `pip` is missing in your venv, bootstrap with:

```bash
python -m ensurepip --upgrade && pip install -r requirements.txt
```

## Dataset Requirements

**IMPORTANT: This project requires the FMA (Free Music Archive) dataset to function.**

### **Required Dataset: FMA Medium**

- **Download**: [FMA Medium Dataset](https://github.com/mdeff/fma#data)
- **Size**: ~22GB (compressed)
- **Contents**: 25,000 tracks with metadata and audio files
- **License**: Creative Commons licenses

### **Setup Instructions**

1. **Download the FMA Medium dataset**:

   ```bash
   # Create data directory
   mkdir -p data
   cd data

   # Download FMA Medium (choose one method)

   # Method 1: Direct download (if available)
   wget https://os.unil.cloud.switch.ch/fma/fma_medium.zip

   # Method 2: Use the FMA download script
   git clone https://github.com/mdeff/fma.git
   cd fma
   python download.py medium
   ```

2. **Extract and organise the dataset**:

   ```bash
   # Extract the dataset
   unzip fma_medium.zip

   # Organise into the expected structure
   mkdir -p data/fma_medium
   mv fma_medium/* data/fma_medium/

   # Download metadata
   wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
   mkdir -p data/fma_metadata
   mv tracks.csv data/fma_metadata/
   ```

3. **Verify dataset structure**:
   ```bash
   # Expected structure
   data/
   ├── fma_medium/           # Audio files (25,000 tracks)
   │   ├── 000/
   │   ├── 001/
   │   └── ...
   └── fma_metadata/
       └── tracks.csv        # Track metadata
   ```

### **Alternative: FMA Small Dataset**

For testing or development, you can use the smaller FMA Small dataset:

- **Size**: ~7.2GB
- **Contents**: 8,000 tracks
- **Download**: Follow the same process but use `python download.py small`

**Note**: Update the `--audio-root` parameter in scripts from `data/fma_medium` to `data/fma_small` when using the small dataset.

## Complete Pipeline (One-Click Solution)

The easiest way to run the entire workflow is using the complete pipeline script. This script executes all steps from track processing to dissertation package generation:

```bash
# Run complete pipeline with default settings
./run_complete_pipeline.sh
```

### **Pipeline Steps**

The script executes 13 steps in sequence:

1. **Data Preparation**: Track selection and audio feature extraction (`robust_track_processing.py`)
2. **Exploratory Data Analysis (EDA)**: Comprehensive dataset analysis (`generate_eda_analysis.py`)
3. **Path Signature Computation**: Compute similarity matrices (`compute_similarity.py`)
4. **Baseline Comparison**: Compare all recommendation methods (`run_baseline_comparison.py`)
5. **Ablation Studies**: Parameter sensitivity analysis (optional, `run_ablation_studies.py`)
6. **Evaluation Suite**: Comprehensive evaluation metrics (`run_evaluation_suite.py`)
7. **Prediction Generation**: Generate genre predictions (`generate_predictions.py`)
8. **Visualisation Generation**: Create all plots and charts (`generate_visualisations.py`)
9. **Dissertation Figures**: Generate publication-quality figures (`generate_dissertation_figures.py`)
10. **Sync Figures**: Copy figures to LaTeX directory (`sync_figures.py`)
11. **Synthetic Users**: Generate synthetic user data for evaluation (optional, `generate_synthetic_users.py`)
12. **Robustness Analysis**: Bootstrap and sensitivity analysis (optional, `run_robustness_analysis.py`)
13. **Dissertation Package**: Package all results for LaTeX (`create_dissertation_package.py`)

### **Pipeline Options**

```bash
# Skip optional steps (ablation studies, synthetic users, robustness analysis)
./run_complete_pipeline.sh --skip-optional

# Resume from a specific step (useful if previous steps already completed)
./run_complete_pipeline.sh --start-from-step 3

# Skip specific steps
./run_complete_pipeline.sh --skip-step 4 --skip-step 10

# Combine options
./run_complete_pipeline.sh --skip-optional --start-from-step 5
```

### **Configuration**

Customise the pipeline using environment variables:

```bash
# Custom track count and seed
N_TRACKS=2000 SEED=2025 ./run_complete_pipeline.sh

# Custom paths and settings
TRACKS_CSV=./data/fma_metadata/tracks.csv \
AUDIO_ROOT=./data/fma_medium \
OUTPUT_DIR=./data/processed_tracks \
SIMILARITY_MATRIX=./data/similarity_matrix.npz \
SIGNATURE_ORDER=2 \
LOG_LEVEL=DEBUG \
./run_complete_pipeline.sh
```

**Available environment variables:**
- `TRACKS_CSV`: Path to tracks.csv metadata file
- `N_TRACKS`: Number of tracks to process (default: 4000)
- `SEED`: Random seed for reproducibility (default: 2025)
- `AUDIO_ROOT`: Root directory for audio files
- `OUTPUT_DIR`: Output directory for processed tracks
- `SIMILARITY_MATRIX`: Path for similarity matrix output
- `N_JOBS`: Number of parallel jobs (default: auto)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `SIGNATURE_ORDER`: Path signature order (default: 1)

### **Pipeline Features**

- **Dependency Checking**: Automatically verifies required files exist before each step
- **Error Handling**: Exits gracefully on critical failures, warns on missing optional dependencies
- **Progress Tracking**: Clear step indicators and completion messages
- **Resume Capability**: Can resume from any step if previous steps completed successfully
- **Flexible Execution**: Skip optional steps or specific steps as needed

## Complete Workflow: Start to Finish


### **Step 1: Data Preparation and Processing**

Select tracks from the FMA dataset and process audio features:

```bash
# Robust track processing with retry mechanism (recommended)
python src/scripts/robust_track_processing.py \
    --tracks-csv data/fma_metadata/tracks.csv \
    --n-tracks 4000 \
    --seed 2025 \
    --audio-root data/fma_medium \
    --output-dir data/processed_tracks \
    --results-dir results/track_processing \
    --log-level INFO
```

This single command handles both track selection and audio processing with automatic retry for failed tracks.

### **Step 2: Exploratory Data Analysis (Optional but Recommended)**

Generate comprehensive EDA to understand your dataset:

```bash
# Generate comprehensive EDA analysis
python src/scripts/generate_eda_analysis.py \
    --tracks-json data/processed_tracks/selected_tracks.json \
    --features-json data/processed_tracks/features.json \
    --output-dir results/eda \
    --dpi 300
```

This generates:
- Genre analysis and distribution

- Temporal analysis
- Outlier detection reports
- Publication-ready visualisations

**Note**: This step is optional but highly recommended for understanding your data before proceeding with analysis.

### **Step 3: Path Signature Computation**

Compute path signatures and similarity matrices:

```bash
# Basic computation (optimal order for best performance/speed ratio)
python src/scripts/compute_similarity.py \
    --features_file data/processed_tracks/features.json \
    --tracks-json data/processed_tracks/selected_tracks.json \
    --output data/similarity_matrix.npz \
    --signature-order 1 \
    --results-dir results/similarity_computation
```

### **Step 4: Baseline Comparison (Recommended)**

Run comprehensive comparison of all methods:

```bash
# Run baseline comparison with optimised path signatures
python src/scripts/run_baseline_comparison.py \
    --features-file data/processed_tracks/features.json \
    --tracks-json data/processed_tracks/selected_tracks.json \
    --output-dir results/baseline_comparison \
    --n-users 200 \
    --test-ratio 0.15 \
    --validation-ratio 0.15
```

### **Step 5: Ablation Studies**

Run comprehensive ablation studies:

```bash
# Run ablation studies
python src/scripts/run_ablation_studies.py \
    --tracks-json data/processed_tracks/selected_tracks.json \
    --features-dir data/processed_tracks \
    --output-dir results/ablation_studies \
    --signature-orders "1,2,3,4" \
    --temperatures "0.1,0.5,1.0,2.0,5.0" \
    --similarity-metrics "cosine,euclidean,manhattan"
```

### **Step 6: Evaluation Suite**

Run comprehensive evaluation:

```bash
# Run evaluation suite
python src/scripts/run_evaluation_suite.py \
    --similarity-matrix data/similarity_matrix.npz \
    --tracks-json data/processed_tracks/selected_tracks.json \
    --output-dir results/evaluation \
    --k-values "1,5,10,20" \
    --cv-folds 5 \
    --test-ratio 0.15 \
    --validation-ratio 0.15 \
    --seed 2025
```

### **Step 7: Generate Predictions for Analysis**

Generate predictions needed for confusion matrix visualisation:

```bash
python src/scripts/generate_predictions.py \
    --features-file data/processed_tracks/features.json \
    --tracks-json data/processed_tracks/selected_tracks.json \
    --output-dir data/predictions.json \
    --signature-order 2 \
    --results-dir results/predictions
```

This generates `predictions.json` with true and predicted genre labels, which is required for confusion matrix generation in the next step.

### **Step 8: Generate Visualisations**

Create visualisations of results:

```bash
# Generate visualisations
python src/scripts/generate_visualisations.py \
    --songs-json data/processed_tracks/selected_tracks.json \
    --similarity-matrix data/similarity_matrix.npz \
    --features-json data/processed_tracks/features.json \
    --predictions-json data/predictions.json \
    --plots-dir results/visualisations
```

This generates:
- Category distribution plots
- Similarity matrix heatmaps
- Feature embeddings (PCA and t-SNE)
- Confusion matrices

### **Step 8.5: Cross-Genre Similarity Analysis (Optional)**

Analyse and visualise songs from different genres with similar acoustic characteristics:

```bash
python src/scripts/find_cross_genre_similar_songs.py \
    --similarity-matrix data/similarity_matrix.npz \
    --tracks-json data/processed_tracks/selected_tracks.json \
    --output results/visualisations/cross_genre_similarity.png \
    --min-similarity 0.7 \
    --top-k 10
```

This generates a comprehensive visualisation (`cross_genre_similarity.png`) including:
- Heatmap showing similarity scores between cross-genre song pairs
- Bar chart of genre combination frequencies
- Histogram of similarity score distribution
- Detailed table of top cross-genre similar song pairs

**Note**: This analysis requires a precomputed similarity matrix from Step 3.

### **Step 9: Generate Dissertation Figures**

Generate publication-quality figures for the dissertation:

```bash
python src/scripts/generate_dissertation_figures.py \
    --output-dir results/dissertation_figures \
    --baseline-results results/baseline_comparison/baseline_comparison_results.json \
    --stats-json results/baseline_comparison/statistical_comparison.json
```

This generates:
- `system_architecture.png`: System architecture flowchart
- `method_comparison.png`: Baseline comparison bar chart with 95% confidence intervals
- `significance_heatmap.png`: Statistical significance heatmap

### **Step 10: Sync Figures for LaTeX**

Sync all figures to the workspace figures directory for LaTeX compilation:

```bash
python src/scripts/sync_figures.py \
    --code-root . \
    --figures-dir ../figures
```

This script copies figures from their generation locations to `/workspace/figures/`:
- `confusion_matrix.png` from `results/evaluation/`
- `feature_embedding_pca.png` and `feature_embedding_tsne.png` from `results/visualisations/`
- `genre_analysis.png` from `results/eda/`
- `user_archetypes.png` and `interaction_heatmap.png` from `results/synthetic_users/`
- `method_comparison.png` from `results/dissertation_figures/`
- `significance_heatmap.png` from `results/dissertation_figures/`
- `system_architecture.png` from `results/dissertation_figures/`
- `baseline_performance.png` from `results/baseline_comparison/`
- `ablation_overview.png` from `results/ablation_studies/`
- `cross_genre_similarity.png` from `results/visualisations/` (if generated)

### **Step 11: Generate Synthetic Users (Optional)**

For evaluation without real user data, generate synthetic users:

```bash
# Generate synthetic users
python src/scripts/generate_synthetic_users.py \
    --tracks-json data/processed_tracks/selected_tracks.json \
    --output-dir results/synthetic_users \
    --n-users 200
```

### **Step 12: Run Robustness Analysis (Optional)**

Perform robustness checks including bootstrap confidence intervals and sensitivity analysis:

```bash
# Run robustness analysis
python src/scripts/run_robustness_analysis.py \
    --output-dir results/robustness \
    --similarity-matrix data/similarity_matrix.npz \
    --tracks-json data/processed_tracks/selected_tracks.json
```

### **Step 13: Path Signature Recommendations**

```bash
python src/scripts/recommendation_main.py \
    --features-file data/processed_tracks/features.json \
    --output-dir results/path_signature \
    --temperature 0.5  # Optimised temperature for better discrimination
```

## Testing

### **Run All Tests**

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_baseline_methods.py -v
pytest tests/test_path_signatures.py -v
pytest tests/test_evaluation_metrics.py -v
pytest tests/test_ablation_studies.py -v
```

## Technical Architecture

### Audio Feature Extraction

- **Multi-dimensional Time Series**: Combines pitch, loudness, tempo, and MFCCs into a unified representation
- **Feature Resampling**: Frame-level features resampled to 10,000 samples for temporal alignment
- **Feature Normalisation**: Normalised features for improved processing
- **Parallel Processing**: Multi-core processing for large datasets

### Path Signatures

- **Mathematical Foundation**: Uses rough path theory to capture musical structure
- **esig Library**: Efficient signature computation via esig library
- **Default Order**: Order 1 (optimal performance/speed ratio); ablations at orders 2–4
- **Genre-based Categories**: Uses actual music metadata instead of arbitrary clustering
- **Enhanced Similarity**: Configurable weights with 80% focus on path signatures

### Softmax Regression

- **Genre-based Categories**: Uses actual genre labels from music metadata
- **Probability Distributions**: Learns probability distributions over categories
- **Optimised Similarity**: Combines multiple similarity metrics with focus on path signatures:
  - Path signature similarity (80% weight) - captures musical shape
  - Category probability similarity (15% weight) - captures genre preferences
  - Category agreement (5% weight) - captures genre matching

### Visualisation

- **Similarity Matrix Heatmap**: Cluster and outlier analysis
- **Category Distribution**: Class balance and bias checking
- **Confusion Matrix**: Model evaluation
- **Feature Embedding (PCA/t-SNE)**: High-dimensional structure visualisation
- **Publication Figures**: Method comparison and significance heatmaps
- **All plots saved to `plots/` directory**; use `sync_figures.py` to copy to workspace for LaTeX

### Temperature Scaling

- **Adjustable Sensitivity**: Temperature parameter controls similarity score sensitivity
- **Optimised Value**: 0.5 provides best discrimination for music recommendation
- **Lower Values**: More pronounced differences between songs
- **Higher Values**: More uniform similarity distribution

### Evaluation Framework

- **Comprehensive Metrics**: Precision@K, Recall@K, NDCG@K, MAP, Diversity, Novelty, Coverage
- **Classification Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC AUC
- **Cross-Validation**: K-fold, Stratified K-fold, Leave-one-out, Recommendation-specific CV
- **Statistical Analysis**: Significance testing and confidence intervals
- **Automated Testing**: Unit tests and validation scripts for reliability

## License

This project is part of an MSc thesis in Computer Science. Please cite appropriately if using this work in your research.
