"""
Data quality reporting utilities for recommendation system evaluation.

This module provides comprehensive data quality analysis and reporting
to help identify issues early and ensure reliable evaluation results.
"""

from typing import Dict, List, Set, Any, Optional
from pathlib import Path
import json
import numpy as np
from ..utils.logger_config import setup_logger
from .types import DataQualityReport

logger = setup_logger("data_quality")


class DataQualityReporter:
    """
    Generate comprehensive data quality reports for recommendation evaluation.
    
    Provides detailed analysis of:
    - ID format consistency
    - Recommendation coverage
    - Ground truth coverage
    - Type distributions
    - Format mismatches
    """
    
    def __init__(self):
        """Initialise the data quality reporter."""
        self.reports = {}
    
    def generate_report(
        self,
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, Set[str]],
        model_name: str,
        output_dir: Optional[Path] = None
    ) -> DataQualityReport:
        """
        Generate comprehensive data quality report.
        
        Args:
            recommendations: Dictionary mapping user_id to list of recommended item IDs
            ground_truth: Dictionary mapping user_id to set of relevant item IDs
            model_name: Name of the model being evaluated
            output_dir: Optional directory to save report JSON file
            
        Returns:
            DataQualityReport dictionary with quality metrics
        """
        report: DataQualityReport = {
            'recommendation_coverage': 0.0,
            'ground_truth_coverage': 0.0,
            'id_format_consistent': True,
            'type_distribution': {},
            'issues': []
        }
        
        # Check recommendation coverage
        total_users = len(recommendations)
        users_with_recs = sum(1 for recs in recommendations.values() if recs)
        report['recommendation_coverage'] = (
            users_with_recs / total_users * 100 if total_users > 0 else 0.0
        )
        
        # Check ground truth coverage
        total_gt_users = len(ground_truth)
        users_with_gt = sum(1 for gt in ground_truth.values() if gt)
        report['ground_truth_coverage'] = (
            users_with_gt / total_gt_users * 100 if total_gt_users > 0 else 0.0
        )
        
        # Collect all IDs for type analysis
        all_rec_ids = []
        for recs in recommendations.values():
            all_rec_ids.extend(recs)
        
        all_gt_ids = []
        for gt in ground_truth.values():
            all_gt_ids.extend(gt)
        
        # Analyse type distribution (sample first 1000 to avoid memory issues)
        sample_size = min(1000, len(all_rec_ids))
        rec_types = {}
        for item_id in all_rec_ids[:sample_size]:
            type_name = type(item_id).__name__
            rec_types[type_name] = rec_types.get(type_name, 0) + 1
        
        sample_size_gt = min(1000, len(all_gt_ids))
        gt_types = {}
        for item_id in list(all_gt_ids)[:sample_size_gt]:
            type_name = type(item_id).__name__
            gt_types[type_name] = gt_types.get(type_name, 0) + 1
        
        report['type_distribution'] = {
            'recommendations': rec_types,
            'ground_truth': gt_types,
            'total_recommendation_ids': len(all_rec_ids),
            'total_ground_truth_ids': len(all_gt_ids),
            'sampled_recommendation_ids': sample_size,
            'sampled_ground_truth_ids': sample_size_gt,
        }
        
        # Check if all are strings
        if set(rec_types.keys()) != {'str'}:
            report['id_format_consistent'] = False
            report['issues'].append(
                f"Recommendation IDs have mixed types: {rec_types}"
            )
        
        if set(gt_types.keys()) != {'str'}:
            report['id_format_consistent'] = False
            report['issues'].append(
                f"Ground truth IDs have mixed types: {gt_types}"
            )
        
        # Check for users with recommendations but no ground truth
        users_with_recs_no_gt = [
            uid for uid in recommendations.keys() 
            if recommendations[uid] and uid not in ground_truth
        ]
        if users_with_recs_no_gt:
            report['issues'].append(
                f"{len(users_with_recs_no_gt)} users have recommendations but no ground truth"
            )
        
        # Check for users with ground truth but no recommendations
        users_with_gt_no_recs = [
            uid for uid in ground_truth.keys() 
            if ground_truth[uid] and (uid not in recommendations or not recommendations[uid])
        ]
        if users_with_gt_no_recs:
            report['issues'].append(
                f"{len(users_with_gt_no_recs)} users have ground truth but no recommendations"
            )
        
        # Calculate overlap statistics
        overlap_stats = self._calculate_overlap_stats(recommendations, ground_truth)
        report['overlap_statistics'] = overlap_stats
        
        # Store report
        self.reports[model_name] = report
        
        # Save to file if output directory provided
        if output_dir:
            self._save_report(report, model_name, output_dir)
        
        # Log summary
        logger.info(
            "%s data quality: Recommendation coverage=%.1f%%, "
            "Ground truth coverage=%.1f%%, ID format consistent=%s, Issues=%d",
            model_name,
            report['recommendation_coverage'],
            report['ground_truth_coverage'],
            report['id_format_consistent'],
            len(report['issues'])
        )
        
        if report['issues']:
            for issue in report['issues']:
                logger.warning("%s: %s", model_name, issue)
        
        return report
    
    def _calculate_overlap_stats(
        self,
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, Set[str]]
    ) -> Dict[str, Any]:
        """Calculate overlap statistics between recommendations and ground truth."""
        overlaps = []
        precisions = []
        
        for user_id in recommendations.keys():
            if user_id not in ground_truth:
                continue
            
            recs = set(recommendations.get(user_id, []))
            gt = ground_truth[user_id]
            
            if not recs or not gt:
                continue
            
            overlap = len(recs & gt)
            overlaps.append(overlap)
            
            if recs:
                precision = overlap / len(recs)
                precisions.append(precision)
        
        return {
            'mean_overlap': float(np.mean(overlaps)) if overlaps else 0.0,
            'median_overlap': float(np.median(overlaps)) if overlaps else 0.0,
            'max_overlap': int(np.max(overlaps)) if overlaps else 0,
            'mean_precision': float(np.mean(precisions)) if precisions else 0.0,
            'users_with_overlap': len(overlaps),
        }
    
    def _save_report(
        self,
        report: DataQualityReport,
        model_name: str,
        output_dir: Path
    ) -> None:
        """Save quality report to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"{model_name}_quality_report.json"
        
        # Convert numpy types to native Python types for JSON serialisation
        json_report = json.loads(json.dumps(report, default=str))
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)
        
        logger.info("Saved quality report to %s", report_file)
    
    def generate_summary_report(self, output_dir: Path) -> None:
        """Generate summary report for all models."""
        summary = {
            'models': list(self.reports.keys()),
            'reports': self.reports,
            'summary_statistics': self._calculate_summary_stats()
        }
        
        summary_file = output_dir / "data_quality_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Saved summary quality report to %s", summary_file)
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics across all models."""
        if not self.reports:
            return {}
        
        coverages = [r['recommendation_coverage'] for r in self.reports.values()]
        gt_coverages = [r['ground_truth_coverage'] for r in self.reports.values()]
        consistent_flags = [r['id_format_consistent'] for r in self.reports.values()]
        
        return {
            'mean_recommendation_coverage': float(np.mean(coverages)),
            'mean_ground_truth_coverage': float(np.mean(gt_coverages)),
            'all_consistent': all(consistent_flags),
            'models_with_issues': sum(
                1 for r in self.reports.values() if r['issues']
            ),
        }

