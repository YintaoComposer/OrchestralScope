"""
Hierarchical clustering and detailed similarity analysis for timbre sequence analysis
"""
from pathlib import Path
from typing import Dict, Any
from .viz import plot_hierarchical_clustering, calculate_detailed_similarities


def create_similarity_analysis_only(results: Dict[str, Any], output_dir: Path) -> None:
    """
    Create only the hierarchical clustering and detailed similarity analysis
    
    Args:
        results: Fine analysis results
        output_dir: Output directory
    """
    print("[timbre-sequence-analyze] Creating similarity analysis only...")
    
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Create hierarchical clustering
    plot_hierarchical_clustering(results, viz_dir)
    
    # Create detailed similarity analysis
    calculate_detailed_similarities(results, output_dir)
    
    print("✓ Similarity analysis generation complete!")


if __name__ == "__main__":
    # This can be used as a standalone script
    import sys
    if len(sys.argv) != 3:
        print("Usage: python similarity_only.py <results_file> <output_dir>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = Path(sys.argv[2])
    
    # Load results (assuming it's a pickle file or similar)
    # This would need to be adapted based on how results are stored
    print("This script is designed to be used within the fine analysis pipeline")