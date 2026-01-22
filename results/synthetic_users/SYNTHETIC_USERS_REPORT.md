# Synthetic Users Generation Report

Generated on: 2026-01-02 12:39:55

## Overview

This report describes the generation of synthetic users for music recommendation system evaluation.

## User Statistics

- **Total Users**: 200
- **Total Interactions**: 80559
- **Sparsity**: 0.899
- **Density**: 0.101

## User Archetype Distribution

- **Casual Listener**: 32.0%
- **Genre Specialist**: 27.0%
- **Music Enthusiast**: 19.0%
- **Mainstream Fan**: 9.0%
- **Explorer**: 13.0%

## Interaction Statistics

- **Mean Interactions per User**: {stats['interaction_stats']['mean']:.1f}
- **Std Interactions per User**: {stats['interaction_stats']['std']:.1f}
- **Min Interactions**: {stats['interaction_stats']['min']}
- **Max Interactions**: {stats['interaction_stats']['max']}
- **Median Interactions**: {stats['interaction_stats']['median']:.1f}

## Rating Distribution

- **Rating 1**: 14823 interactions (18.4%)
- **Rating 2**: 44784 interactions (55.6%)
- **Rating 3**: 20892 interactions (25.9%)
- **Rating 4**: 60 interactions (0.1%)

## Genre Distribution

- **Rock**: 40135 interactions (49.8%)
- **Electronic**: 14389 interactions (17.9%)
- **Hip-Hop**: 4923 interactions (6.1%)
- **Classical**: 3667 interactions (4.6%)
- **Jazz**: 3188 interactions (4.0%)
- **Pop**: 3077 interactions (3.8%)
- **Experimental**: 2160 interactions (2.7%)
- **Folk**: 1756 interactions (2.2%)
- **Soul-RnB**: 1530 interactions (1.9%)
- **Old-Time / Historic**: 1335 interactions (1.7%)

## Generated Files

### Data Files
- `synthetic_users.json`: Complete user profiles with preferences
- `user_interactions.json`: User-item interaction matrix
- `user_statistics.json`: Comprehensive user statistics
- `user_statistics_table.csv`: LaTeX-ready user statistics table
- `interaction_matrix_summary.json`: Matrix statistics (sparsity, density)

### Visualizations
- `user_archetypes.png`: User archetype distribution (300 DPI)
- `interaction_heatmap.png`: User-item interaction heatmap (300 DPI)

## Methodology

### User Archetypes

The synthetic users are generated using five distinct archetypes:

1. **Music Enthusiast**: High engagement, diverse preferences, moderate novelty seeking
2. **Genre Specialist**: Focused on specific genres, moderate engagement
3. **Casual Listener**: Low engagement, popular music focus, low diversity
4. **Explorer**: High novelty seeking, diverse discovery, high engagement
5. **Mainstream Fan**: Popular music focus, low diversity, moderate engagement

### Interaction Generation

User interactions are generated based on:
- **Engagement Level**: Determines number of interactions
- **Diversity Preference**: Influences genre variety
- **Novelty Seeking**: Affects discovery of new content
- **Popularity Bias**: Influences preference for popular content
- **Genre Preferences**: Directs interactions toward preferred genres

### Realistic Patterns

The generation process ensures:
- Realistic sparsity patterns (typical of real user data)
- Balanced archetype distribution
- Appropriate interaction rates per archetype
- Genre preference alignment
- Rating distribution patterns

## Usage

These synthetic users can be used for:
- Recommendation system evaluation
- Cross-validation experiments
- A/B testing of algorithms
- Performance benchmarking
- Diversity and novelty analysis

## Files Generated

This analysis generates 8 dissertation-ready output files suitable for inclusion in MSc dissertation methodology and results chapters.
