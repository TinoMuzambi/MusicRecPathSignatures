"""
Generate system architecture diagram for the dissertation.

This script creates a flowchart showing the complete data flow from raw audio
through feature extraction, path signature computation, and recommendation generation.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

from src.utils.logger_config import setup_logger, configure_logging


logger = setup_logger("generate_system_architecture")


def set_publication_style():
    """Set publication-quality styling."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    })


def create_system_architecture_diagram(output_path: Path, dpi: int = 300):
    """
    Create system architecture diagram showing data flow.
    
    Args:
        output_path: Path to save the figure
        dpi: Resolution for saved figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define component positions and sizes
    # Top row: Input (moved down to avoid title)
    input_box = {'x': 1, 'y': 7.8, 'width': 1.5, 'height': 0.8}
    
    # Middle row: Processing components (moved down)
    preprocess_box = {'x': 0.5, 'y': 5.8, 'width': 1.8, 'height': 0.8}
    feature_box = {'x': 3, 'y': 5.8, 'width': 1.8, 'height': 0.8}
    path_box = {'x': 5.5, 'y': 5.8, 'width': 1.8, 'height': 0.8}
    signature_box = {'x': 8, 'y': 5.8, 'width': 1.8, 'height': 0.8}
    
    # Bottom row: Output components (moved down)
    similarity_box = {'x': 3, 'y': 3.3, 'width': 1.8, 'height': 0.8}
    recommendation_box = {'x': 5.5, 'y': 3.3, 'width': 1.8, 'height': 0.8}
    
    # Bottom: Output (moved down)
    output_box = {'x': 4.25, 'y': 0.8, 'width': 1.5, 'height': 0.8}
    
    # Helper function to create rounded boxes
    def create_box(x, y, width, height, label, color='lightblue', text_color='black'):
        """Create a rounded rectangle box with text."""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=1.5,
            zorder=2  # Boxes at z-order 2
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(
            x + width/2, y + height/2,
            label,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color=text_color,
            zorder=4  # Text on top of boxes
        )
        return box
    
    # Helper function to create arrows
    def create_arrow(x1, y1, x2, y2, style='->', color='black', linewidth=1.5):
        """Create an arrow between two points."""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            color=color,
            linewidth=linewidth,
            zorder=3,  # Arrows at z-order 3 (above boxes, below text)
            mutation_scale=20
        )
        ax.add_patch(arrow)
        return arrow
    
    # Create input
    create_box(**input_box, label='Raw Audio\nFiles', color='lightgreen')
    
    # Create processing components
    create_box(**preprocess_box, label='Audio\nPreprocessing', color='lightblue')
    create_box(**feature_box, label='Feature\nExtraction', color='lightblue')
    create_box(**path_box, label='Path\nConstruction', color='lightblue')
    create_box(**signature_box, label='Signature\nComputation', color='lightblue')
    
    # Create output components
    create_box(**similarity_box, label='Similarity\nCalculation', color='lightcoral')
    create_box(**recommendation_box, label='Recommendation\nGeneration', color='lightcoral')
    
    # Create output
    create_box(**output_box, label='Recommendations', color='lightyellow')
    
    # Create arrows showing data flow
    # Input to preprocessing
    create_arrow(
        input_box['x'] + input_box['width']/2,
        input_box['y'],
        preprocess_box['x'] + preprocess_box['width']/2,
        preprocess_box['y'] + preprocess_box['height']
    )
    
    # Preprocessing to feature extraction
    create_arrow(
        preprocess_box['x'] + preprocess_box['width'],
        preprocess_box['y'] + preprocess_box['height']/2,
        feature_box['x'],
        feature_box['y'] + feature_box['height']/2
    )
    
    # Feature extraction to path construction
    create_arrow(
        feature_box['x'] + feature_box['width'],
        feature_box['y'] + feature_box['height']/2,
        path_box['x'],
        path_box['y'] + path_box['height']/2
    )
    
    # Path construction to signature computation
    create_arrow(
        path_box['x'] + path_box['width'],
        path_box['y'] + path_box['height']/2,
        signature_box['x'],
        signature_box['y'] + signature_box['height']/2
    )
    
    # Signature computation to similarity calculation
    create_arrow(
        signature_box['x'] + signature_box['width']/2,
        signature_box['y'],
        similarity_box['x'] + similarity_box['width']/2,
        similarity_box['y'] + similarity_box['height']
    )
    
    # Similarity calculation to recommendation generation
    create_arrow(
        similarity_box['x'] + similarity_box['width'],
        similarity_box['y'] + similarity_box['height']/2,
        recommendation_box['x'],
        recommendation_box['y'] + recommendation_box['height']/2
    )
    
    # Recommendation generation to output
    create_arrow(
        recommendation_box['x'] + recommendation_box['width']/2,
        recommendation_box['y'],
        output_box['x'] + output_box['width']/2,
        output_box['y'] + output_box['height']
    )
    
    # Add title
    ax.text(
        5, 9.5,
        'System Architecture: Path Signature-Based Music Recommendation',
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        zorder=4
    )
    
    # Add component descriptions on the left side (moved lower to avoid intersection)
    descriptions = [
        "• Resampling to 22.05 kHz\n• Peak normalisation\n• Quality checks",
        "• MFCCs (13)\n• Spectral features\n• Chroma (12)\n• Temporal features",
        "• Multi-dimensional\n  time series\n• Fixed length (10k)\n• Standardisation",
        "• esig library\n• Truncation order 1\n• L2 normalisation",
        "• Cosine similarity\n• L2-normalised\n  signatures",
        "• Top-K ranking\n• Similarity-based\n  retrieval"
    ]
    
    desc_x = 0.2
    desc_y_start = 2.8  # Moved lower to avoid intersection with diagram
    desc_spacing = 0.9
    
    for i, desc in enumerate(descriptions):
        y_pos = desc_y_start - i * desc_spacing
        ax.text(
            desc_x, y_pos,
            desc,
            ha='left', va='top',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'),
            zorder=1  # Lower z-order so arrows appear on top
        )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logger.info("System architecture diagram saved to %s", output_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate system architecture diagram"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="code/results/dissertation_figures/system_architecture.png",
        help="Output path for the diagram"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure (default: 300)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Determine output directory for log file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with file handler
    log_file = output_path.parent / "generate_system_architecture.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger.info("Logging to file: %s", log_file)
    
    logger.info("Generating system architecture diagram...")
    create_system_architecture_diagram(output_path, dpi=args.dpi)
    logger.info("Diagram generated successfully at %s", output_path)


if __name__ == "__main__":
    main()

