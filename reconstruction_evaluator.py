import os
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import our reconstruction modules
from main_pipeline import ReconstructionPipeline
from config_manager import ConfigurationManager, ReconstructionConfig

@dataclass
class BatchJob:
    """Configuration for a single batch reconstruction job"""
    job_id: str
    input_directory: str
    output_directory: str
    config: ReconstructionConfig
    description: str = ""
    priority: int = 1  # Higher numbers = higher priority
    
class BatchProcessor:
    """
    Batch processing system for multiple reconstruction jobs
    """
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.job_queue: List[BatchJob] = []
        self.completed_jobs: List[Dict] = []
        self.failed_jobs: List[Dict] = []
        self.config_manager = ConfigurationManager()
    
    def add_job(self, job: BatchJob):
        """Add a job to the processing queue"""
        self.job_queue.append(job)
        print(f"Added job: {job.job_id}")
    
    def add_comparative_jobs(self, input_directory: str, base_output_dir: str, 
                           configs: Dict[str, ReconstructionConfig], 
                           description_prefix: str = ""):
        """
        Add multiple jobs to compare different configurations on the same dataset
        
        Args:
            input_directory: Input data directory
            base_output_dir: Base output directory
            configs: Dictionary of {config_name: config} pairs
            description_prefix: Prefix for job descriptions
        """
        base_output_path = Path(base_output_dir)
        
        for config_name, config in configs.items():
            job_id = f"{description_prefix}_{config_name}_{int(time.time())}"
            output_dir = base_output_path / config_name
            
            job = BatchJob(
                job_id=job_id,
                input_directory=input_directory,
                output_directory=str(output_dir),
                config=config,
                description=f"{description_prefix} with {config_name} configuration"
            )
            
            self.add_job(job)
    
    def run_single_job(self, job: BatchJob) -> Dict:
        """
        Run a single reconstruction job
        
        Returns:
            Dictionary with job results
        """
        start_time = time.time()
        
        job_result = {
            'job_id': job.job_id,
            'description': job.description,
            'input_directory': job.input_directory,
            'output_directory': job.output_directory,
            'start_time': start_time,
            'end_time': None,
            'duration': None,
            'success': False,
            'error_message': None,
            'metrics': {},
            'config_used': asdict(job.config)
        }
        
        try:
            print(f"Starting job: {job.job_id}")
            
            # Create output directory
            Path(job.output_directory).mkdir(parents=True, exist_ok=True)
            
            # Save job configuration
            config_path = Path(job.output_directory) / "job_config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(job.config), f, indent=2)
            
            # Run reconstruction pipeline
            pipeline = ReconstructionPipeline(asdict(job.config))
            success = pipeline.run_full_pipeline(job.input_directory, job.output_directory)
            
            if success:
                job_result['success'] = True
                job_result['metrics'] = pipeline.quality_metrics
                
                # Extract additional metrics
                job_result['num_fragments'] = len(pipeline.fragments)
                job_result['num_aligned_fragments'] = len(pipeline.transformations)
                job_result['total_matches'] = sum(
                    len(matches) for color_matches in pipeline.all_matches.values()
                    for matches in color_matches.values()
                )
                
                print(f"Job {job.job_id} completed successfully")
            else:
                job_result['error_message'] = "Pipeline failed"
                print(f"Job {job.job_id} failed")
                
        except Exception as e:
            job_result['error_message'] = str(e)
            print(f"Job {job.job_id} failed with exception: {e}")
        
        finally:
            end_time = time.time()
            job_result['end_time'] = end_time
            job_result['duration'] = end_time - start_time
        
        return job_result
    
    def run_batch_sequential(self) -> List[Dict]:
        """Run all jobs sequentially"""
        print(f"Starting batch processing of {len(self.job_queue)} jobs (sequential)")
        
        # Sort jobs by priority
        sorted_jobs = sorted(self.job_queue, key=lambda x: x.priority, reverse=True)
        
        results = []
        
        for i, job in enumerate(sorted_jobs):
            print(f"\nProcessing job {i+1}/{len(sorted_jobs)}: {job.job_id}")
            
            result = self.run_single_job(job)
            results.append(result)
            
            if result['success']:
                self.completed_jobs.append(result)
            else:
                self.failed_jobs.append(result)
        
        self.job_queue.clear()
        return results
    
    def run_batch_parallel(self) -> List[Dict]:
        """Run jobs in parallel using multiprocessing"""
        print(f"Starting batch processing of {len(self.job_queue)} jobs (parallel, {self.max_workers} workers)")
        
        # Sort jobs by priority
        sorted_jobs = sorted(self.job_queue, key=lambda x: x.priority, reverse=True)
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(self.run_single_job, job): job 
                            for job in sorted_jobs}
            
            # Collect results as they complete
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        self.completed_jobs.append(result)
                        print(f"✓ Job {job.job_id} completed successfully")
                    else:
                        self.failed_jobs.append(result)
                        print(f"✗ Job {job.job_id} failed")
                        
                except Exception as e:
                    error_result = {
                        'job_id': job.job_id,
                        'success': False,
                        'error_message': f"Execution error: {str(e)}"
                    }
                    results.append(error_result)
                    self.failed_jobs.append(error_result)
                    print(f"✗ Job {job.job_id} failed with execution error")
        
        self.job_queue.clear()
        return results
    
    def save_batch_results(self, results: List[Dict], output_path: str):
        """Save batch processing results to JSON"""
        batch_summary = {
            'batch_info': {
                'total_jobs': len(results),
                'successful_jobs': len(self.completed_jobs),
                'failed_jobs': len(self.failed_jobs),
                'total_duration': sum(r.get('duration', 0) for r in results),
                'timestamp': time.time()
            },
            'job_results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"Batch results saved to: {output_path}")

class ReconstructionEvaluator:
    """
    Evaluate and compare reconstruction results
    """
    
    def __init__(self):
        pass
    
    def load_batch_results(self, results_file: str) -> Dict:
        """Load batch processing results from JSON"""
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def compare_configurations(self, batch_results: Dict, output_dir: str):
        """
        Compare different configurations on the same dataset
        
        Args:
            batch_results: Results from batch processing
            output_dir: Directory to save comparison reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract successful jobs
        successful_jobs = [job for job in batch_results['job_results'] if job['success']]
        
        if len(successful_jobs) < 2:
            print("Need at least 2 successful jobs for comparison")
            return
        
        # Group jobs by input directory (same dataset)
        dataset_groups = {}
        for job in successful_jobs:
            input_dir = job['input_directory']
            if input_dir not in dataset_groups:
                dataset_groups[input_dir] = []
            dataset_groups[input_dir].append(job)
        
        # Compare each dataset group
        for dataset_path, jobs in dataset_groups.items():
            if len(jobs) < 2:
                continue
            
            dataset_name = Path(dataset_path).name
            print(f"Comparing configurations for dataset: {dataset_name}")
            
            self._compare_dataset_jobs(jobs, output_dir / f"comparison_{dataset_name}")
    
    def _compare_dataset_jobs(self, jobs: List[Dict], output_dir: Path):
        """Compare jobs on the same dataset"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract metrics for comparison
        comparison_data = []
        
        for job in jobs:
            config_name = self._extract_config_name(job)
            metrics = job.get('metrics', {})
            
            row = {
                'job_id': job['job_id'],
                'config_name': config_name,
                'duration': job.get('duration', 0),
                'num_fragments': job.get('num_fragments', 0),
                'num_aligned_fragments': job.get('num_aligned_fragments', 0),
                'total_matches': job.get('total_matches', 0),
                'alignment_success_rate': (job.get('num_aligned_fragments', 0) / 
                                         max(job.get('num_fragments', 1), 1)),
                **metrics  # Add all quality metrics
            }
            
            comparison_data.append(row)
        
        # Create DataFrame for analysis
        df = pd.DataFrame(comparison_data)
        
        # Save detailed comparison table
        df.to_csv(output_dir / "detailed_comparison.csv", index=False)
        
        # Create comparison plots
        self._create_comparison_plots(df, output_dir)
        
        # Generate summary report
        self._generate_comparison_report(df, output_dir)
    
    def _extract_config_name(self, job: Dict) -> str:
        """Extract configuration name from job description or ID"""
        desc = job.get('description', '')
        job_id = job.get('job_id', '')
        
        # Try to extract config name from description or job_id
        for name in ['default', 'high_quality', 'fast_processing', 'conservative', 'aggressive']:
            if name in desc.lower() or name in job_id.lower():
                return name
        
        # Fallback to job_id
        return job_id.split('_')[0] if '_' in job_id else job_id
    
    def _create_comparison_plots(self, df: pd.DataFrame, output_dir: Path):
        """Create comparison plots"""
        
        # Plot 1: Processing time vs quality
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Configuration Comparison', fontsize=16)
        
        # Duration comparison
        ax = axes[0, 0]
        df.plot(x='config_name', y='duration', kind='bar', ax=ax, color='skyblue')
        ax.set_title('Processing Duration by Configuration')
        ax.set_ylabel('Duration (seconds)')
        ax.tick_params(axis='x', rotation=45)
        
        # Alignment success rate
        ax = axes[0, 1]
        df.plot(x='config_name', y='alignment_success_rate', kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Alignment Success Rate')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Number of matches found
        ax = axes[1, 0]
        df.plot(x='config_name', y='total_matches', kind='bar', ax=ax, color='orange')
        ax.set_title('Total Matches Found')
        ax.set_ylabel('Number of Matches')
        ax.tick_params(axis='x', rotation=45)
        
        # Quality metric (if available)
        if 'overall_mean_error' in df.columns:
            ax = axes[1, 1]
            df.plot(x='config_name', y='overall_mean_error', kind='bar', ax=ax, color='salmon')
            ax.set_title('Mean Alignment Error')
            ax.set_ylabel('Error (distance units)')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "configuration_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Radar chart for multi-metric comparison
        self._create_radar_chart(df, output_dir)
    
    def _create_radar_chart(self, df: pd.DataFrame, output_dir: Path):
        """Create radar chart for multi-metric comparison"""
        try:
            # Select metrics for radar chart
            metrics = ['alignment_success_rate', 'total_matches']
            
            # Add normalized duration (inverted, so faster is better)
            max_duration = df['duration'].max()
            df['speed_score'] = 1 - (df['duration'] / max_duration)
            metrics.append('speed_score')
            
            # Add quality score if available
            if 'overall_mean_error' in df.columns:
                # Invert error so lower error = higher score
                max_error = df['overall_mean_error'].max()
                df['quality_score'] = 1 - (df['overall_mean_error'] / max_error)
                metrics.append('quality_score')
            
            # Normalize all metrics to 0-1 scale
            df_norm = df.copy()
            for metric in metrics:
                if metric not in ['alignment_success_rate', 'speed_score', 'quality_score']:
                    df_norm[metric] = df[metric] / df[metric].max()
            
            # Create radar chart
            from math import pi
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
            angles += angles[:1]  # Complete the circle
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
            
            for i, (_, row) in enumerate(df_norm.iterrows()):
                values = [row[metric] for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=row['config_name'], color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('Configuration Performance Radar Chart', y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            plt.savefig(output_dir / "radar_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Could not create radar chart: {e}")
    
    def _generate_comparison_report(self, df: pd.DataFrame, output_dir: Path):
        """Generate a text report comparing configurations"""
        
        report = []
        report.append("CONFIGURATION COMPARISON REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY:")
        report.append(f"Configurations compared: {len(df)}")
        report.append(f"Average processing time: {df['duration'].mean():.2f} seconds")
        report.append(f"Average alignment rate: {df['alignment_success_rate'].mean():.2%}")
        report.append(f"Average matches found: {df['total_matches'].mean():.1f}")
        report.append("")
        
        # Best performers
        report.append("BEST PERFORMERS:")
        
        fastest = df.loc[df['duration'].idxmin()]
        report.append(f"Fastest: {fastest['config_name']} ({fastest['duration']:.2f}s)")
        
        best_alignment = df.loc[df['alignment_success_rate'].idxmax()]
        report.append(f"Best alignment rate: {best_alignment['config_name']} ({best_alignment['alignment_success_rate']:.2%})")
        
        most_matches = df.loc[df['total_matches'].idxmax()]
        report.append(f"Most matches: {most_matches['config_name']} ({most_matches['total_matches']} matches)")
        
        if 'overall_mean_error' in df.columns:
            best_quality = df.loc[df['overall_mean_error'].idxmin()]
            report.append(f"Best quality: {best_quality['config_name']} (error: {best_quality['overall_mean_error']:.4f})")
        
        report.append("")
        
        # Detailed breakdown
        report.append("DETAILED RESULTS:")
        for _, row in df.iterrows():
            report.append(f"\n{row['config_name']}:")
            report.append(f"  Duration: {row['duration']:.2f}s")
            report.append(f"  Fragments aligned: {row['num_aligned_fragments']}/{row['num_fragments']}")
            report.append(f"  Alignment rate: {row['alignment_success_rate']:.2%}")
            report.append(f"  Total matches: {row['total_matches']}")
            
            if 'overall_mean_error' in row and pd.notna(row['overall_mean_error']):
                report.append(f"  Mean error: {row['overall_mean_error']:.4f}")
        
        # Save report
        with open(output_dir / "comparison_report.txt", 'w') as f:
            f.write("\n".join(report))
        
        print(f"Comparison report saved to: {output_dir / 'comparison_report.txt'}")

# CLI interface for batch processing
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Processing for Stele Reconstruction")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Batch run command
    batch_parser = subparsers.add_parser('batch', help='Run batch reconstruction jobs')
    batch_parser.add_argument('config_file', help='JSON file with batch job configurations')
    batch_parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    batch_parser.add_argument('--parallel', action='store_true', help='Run jobs in parallel')
    batch_parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    
    # Compare configurations command
    compare_parser = subparsers.add_parser('compare', help='Compare different configurations')
    compare_parser.add_argument('input_data', help='Directory with PLY files')
    compare_parser.add_argument('output_dir', help='Output directory for comparison')
    compare_parser.add_argument('--configs', nargs='+', 
                               choices=['default', 'high_quality', 'fast_processing', 'conservative', 'aggressive'],
                               default=['default', 'high_quality', 'fast_processing'],
                               help='Configurations to compare')
    compare_parser.add_argument('--parallel', action='store_true', help='Run comparisons in parallel')
    
    # Evaluate results command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate batch results')
    evaluate_parser.add_argument('results_file', help='JSON file with batch results')
    evaluate_parser.add_argument('output_dir', help='Output directory for evaluation reports')
    
    args = parser.parse_args()
    
    if args.command == 'batch':
        # Load job configurations
        with open(args.config_file, 'r') as f:
            batch_config = json.load(f)
        
        processor = BatchProcessor(max_workers=args.workers)
        config_manager = ConfigurationManager()
        
        # Create jobs from configuration
        for job_config in batch_config['jobs']:
            # Load reconstruction config
            if 'config_template' in job_config:
                recon_config = config_manager.get_template_config(job_config['config_template'])
            else:
                recon_config = config_manager._dict_to_config(job_config['config'])
            
            job = BatchJob(
                job_id=job_config['job_id'],
                input_directory=job_config['input_directory'],
                output_directory=job_config['output_directory'],
                config=recon_config,
                description=job_config.get('description', ''),
                priority=job_config.get('priority', 1)
            )
            
            processor.add_job(job)
        
        # Run batch
        if args.parallel:
            results = processor.run_batch_parallel()
        else:
            results = processor.run_batch_sequential()
        
        # Save results
        output_path = Path(args.output) / "batch_results.json"
        processor.save_batch_results(results, str(output_path))
        
        print(f"\nBatch processing completed:")
        print(f"  Total jobs: {len(results)}")
        print(f"  Successful: {len(processor.completed_jobs)}")
        print(f"  Failed: {len(processor.failed_jobs)}")
    
    elif args.command == 'compare':
        processor = BatchProcessor(max_workers=args.workers if args.parallel else 1)
        config_manager = ConfigurationManager()
        
        # Create comparison jobs
        configs = {name: config_manager.get_template_config(name) for name in args.configs}
        
        processor.add_comparative_jobs(
            args.input_data,
            args.output_dir,
            configs,
            "config_comparison"
        )
        
        # Run comparisons
        if args.parallel:
            results = processor.run_batch_parallel()
        else:
            results = processor.run_batch_sequential()
        
        # Save results
        results_file = Path(args.output_dir) / "comparison_results.json"
        processor.save_batch_results(results, str(results_file))
        
        # Generate comparison report
        evaluator = ReconstructionEvaluator()
        batch_data = evaluator.load_batch_results(str(results_file))
        evaluator.compare_configurations(batch_data, args.output_dir)
    
    elif args.command == 'evaluate':
        evaluator = ReconstructionEvaluator()
        batch_results = evaluator.load_batch_results(args.results_file)
        evaluator.compare_configurations(batch_results, args.output_dir)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()