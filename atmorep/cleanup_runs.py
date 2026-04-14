#!/usr/bin/env python3
"""
Cleanup script for failed/unwanted AtmoRep evaluation runs.

This script moves output files, logs, wandb runs, results, and models
associated with specified evaluation rounds or job IDs to a trash folder.

Usage:
    python cleanup_runs.py --round 3                    # Clean ROUND3
    python cleanup_runs.py --round 3 --dry-run          # Preview what would be cleaned
    python cleanup_runs.py --job-ids 21584933,21584934  # Clean specific jobs
    python cleanup_runs.py --round 2 --round 3          # Clean multiple rounds
    python cleanup_runs.py --list-rounds                # Show available rounds
"""

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class RunCleaner:
    def __init__(self, base_dir: str = "/work/ab1412/atmorep", dry_run: bool = False):
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        self.ids_file = self.base_dir / "results" / "important_result_ids.txt"
        
        # Counters
        self.stats = {
            'output': 0,
            'logs': 0,
            'wandb': 0,
            'results': 0,
            'models': 0,
        }
        
        self.trash_dir: Optional[Path] = None
        self.wandb_ids_found: set = set()
    
    def parse_ids_file(self) -> dict:
        """Parse important_result_ids.txt to extract job IDs for each round."""
        rounds = {}
        current_round = None
        
        if not self.ids_file.exists():
            print(f"Warning: {self.ids_file} not found")
            return rounds
        
        with open(self.ids_file, 'r') as f:
            content = f.read()
        
        # Find ROUND sections and their job IDs
        # Look for patterns like "## ROUND2 ##"
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Detect round headers
            round_match = re.search(r'ROUND\s*(\d+)', line, re.IGNORECASE)
            if round_match:
                current_round = int(round_match.group(1))
                if current_round not in rounds:
                    rounds[current_round] = {'job_ids': [], 'wandb_ids': []}
            
            # Extract SLURM job IDs (8-digit numbers in job listings)
            if current_round is not None:
                # Match job IDs from squeue output (format: 20466838 or 21584933)
                job_matches = re.findall(r'\b(2\d{7})\b', line)
                for job_id in job_matches:
                    if job_id not in rounds[current_round]['job_ids']:
                        rounds[current_round]['job_ids'].append(job_id)
                
                # Match wandb IDs (8-character alphanumeric, from lists)
                wandb_matches = re.findall(r"'([a-z0-9]{8})'", line)
                for wid in wandb_matches:
                    if wid not in rounds[current_round]['wandb_ids']:
                        rounds[current_round]['wandb_ids'].append(wid)
        
        return rounds
    
    def list_rounds(self):
        """List available rounds and their job counts."""
        rounds = self.parse_ids_file()
        
        print("\n=== Available Rounds in important_result_ids.txt ===\n")
        
        if not rounds:
            print("No rounds found in the IDs file.")
            return
        
        for round_num in sorted(rounds.keys()):
            info = rounds[round_num]
            print(f"ROUND {round_num}:")
            print(f"  Job IDs:   {len(info['job_ids'])} jobs")
            if info['job_ids']:
                print(f"             First: {info['job_ids'][0]}, Last: {info['job_ids'][-1]}")
            print(f"  Wandb IDs: {len(info['wandb_ids'])} runs")
            print()
    
    def create_trash_dir(self, round_nums: list):
        """Create timestamped trash directory."""
        round_str = "_".join([f"round{r}" for r in round_nums])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trash_dir = self.base_dir / "trash" / f"{round_str}_{timestamp}"
        
        if not self.dry_run:
            for subdir in ['output', 'logs', 'wandb', 'results', 'models']:
                (self.trash_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        return self.trash_dir
    
    def move_file(self, src: Path, dest_subdir: str) -> bool:
        """Move a file to trash directory."""
        if not src.exists():
            return False
        
        dest = self.trash_dir / dest_subdir / src.name
        
        if self.dry_run:
            print(f"  [DRY-RUN] Would move: {src.name}")
            return True
        else:
            try:
                shutil.move(str(src), str(dest))
                return True
            except Exception as e:
                print(f"  [ERROR] Failed to move {src.name}: {e}")
                return False
    
    def move_directory(self, src: Path, dest_subdir: str) -> bool:
        """Move a directory to trash."""
        if not src.exists():
            return False
        
        dest = self.trash_dir / dest_subdir / src.name
        
        if self.dry_run:
            print(f"  [DRY-RUN] Would move dir: {src.name}")
            return True
        else:
            try:
                shutil.move(str(src), str(dest))
                return True
            except Exception as e:
                print(f"  [ERROR] Failed to move {src.name}: {e}")
                return False
    
    def clean_output_files(self, job_ids: list):
        """Move output files for given job IDs."""
        print("\nCleaning output files...")
        output_dir = self.base_dir / "output"
        
        for job_id in job_ids:
            output_file = output_dir / f"output_{job_id}.txt"
            # Extract wandb IDs from this file for later use
            if output_file.exists() and not self.dry_run:
                self._extract_wandb_ids(output_file)
            if self.move_file(output_file, 'output'):
                self.stats['output'] += 1
        
        # For dry run, extract wandb IDs before "moving"
        if self.dry_run:
            for job_id in job_ids:
                output_file = output_dir / f"output_{job_id}.txt"
                if output_file.exists():
                    self._extract_wandb_ids(output_file)
        
        print(f"  Found {self.stats['output']} output files")
    
    def _extract_wandb_ids(self, output_file: Path):
        """Extract wandb run IDs from output file."""
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            # Match wandb IDs like "atmorep-t94gxdjq-21584935"
            matches = re.findall(r'atmorep-([a-z0-9]{8})-\d+', content)
            self.wandb_ids_found.update(matches)
        except Exception:
            pass
    
    def clean_log_files(self, job_ids: list):
        """Move log files for given job IDs."""
        print("\nCleaning log files...")
        logs_dir = self.base_dir / "logs"
        
        for job_id in job_ids:
            # Try various naming patterns
            patterns = [
                f"nice_eval_round*_{job_id}.*",
                f"nice_eval_{job_id}.*",
                f"nice_batch_wrapper{job_id}.*",
                f"*{job_id}*.err",
                f"*{job_id}*.out",
            ]
            
            for pattern in patterns:
                for log_file in logs_dir.glob(pattern):
                    if self.move_file(log_file, 'logs'):
                        self.stats['logs'] += 1
        
        print(f"  Found {self.stats['logs']} log files")
    
    def clean_wandb_runs(self, job_ids: list, wandb_ids: list):
        """Move wandb runs associated with job IDs or wandb IDs."""
        print("\nCleaning wandb runs...")
        wandb_dir = self.base_dir / "wandb"
        
        if not wandb_dir.exists():
            print("  No wandb directory found")
            return
        
        # Combine provided wandb_ids with ones found in output files
        all_wandb_ids = set(wandb_ids) | self.wandb_ids_found
        
        for run_dir in wandb_dir.glob("offline-run-*"):
            if not run_dir.is_dir():
                continue
            
            should_move = False
            
            # Check if run contains any of our job IDs
            for job_id in job_ids:
                # Check config.yaml for slurm_job_id
                config_file = run_dir / "files" / "config.yaml"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            if job_id in f.read():
                                should_move = True
                                break
                    except Exception:
                        pass
            
            # Check if run name contains any wandb ID
            if not should_move:
                for wid in all_wandb_ids:
                    if wid in run_dir.name:
                        should_move = True
                        break
            
            if should_move:
                if self.move_directory(run_dir, 'wandb'):
                    self.stats['wandb'] += 1
        
        print(f"  Found {self.stats['wandb']} wandb runs")
    
    def clean_results(self, wandb_ids: list):
        """Move result files matching wandb IDs."""
        print("\nCleaning result files...")
        results_dir = self.base_dir / "results"
        
        all_wandb_ids = set(wandb_ids) | self.wandb_ids_found
        
        for wid in all_wandb_ids:
            for result_file in results_dir.glob(f"*{wid}*"):
                if result_file.is_file():
                    if self.move_file(result_file, 'results'):
                        self.stats['results'] += 1
                elif result_file.is_dir():
                    if self.move_directory(result_file, 'results'):
                        self.stats['results'] += 1
        
        print(f"  Found {self.stats['results']} result files/dirs")
    
    def clean_models(self, wandb_ids: list):
        """Move model files matching wandb IDs."""
        print("\nCleaning model files...")
        models_dir = self.base_dir / "models"
        
        if not models_dir.exists():
            print("  No models directory found")
            return
        
        all_wandb_ids = set(wandb_ids) | self.wandb_ids_found
        
        for wid in all_wandb_ids:
            for model_file in models_dir.glob(f"*{wid}*"):
                if model_file.is_file():
                    if self.move_file(model_file, 'models'):
                        self.stats['models'] += 1
                elif model_file.is_dir():
                    if self.move_directory(model_file, 'models'):
                        self.stats['models'] += 1
        
        print(f"  Found {self.stats['models']} model files/dirs")
    
    def print_summary(self):
        """Print cleanup summary."""
        print("\n" + "=" * 50)
        print("CLEANUP SUMMARY")
        print("=" * 50)
        
        if self.dry_run:
            print("[DRY-RUN MODE - No files were actually moved]\n")
        
        print(f"Output files:  {self.stats['output']}")
        print(f"Log files:     {self.stats['logs']}")
        print(f"Wandb runs:    {self.stats['wandb']}")
        print(f"Result files:  {self.stats['results']}")
        print(f"Model files:   {self.stats['models']}")
        print(f"\nTotal items:   {sum(self.stats.values())}")
        
        if not self.dry_run and self.trash_dir:
            print(f"\nAll files moved to:\n  {self.trash_dir}")
            print(f"\nTo permanently delete, run:")
            print(f"  rm -rf {self.trash_dir}")
            print(f"\nTo restore, move files back from trash subfolders")
        elif self.dry_run:
            print(f"\nTo actually clean, run without --dry-run")
    
    def clean_rounds(self, round_nums: list):
        """Clean specified rounds."""
        rounds = self.parse_ids_file()
        
        all_job_ids = []
        all_wandb_ids = []
        
        for rnum in round_nums:
            if rnum not in rounds:
                print(f"Warning: ROUND {rnum} not found in {self.ids_file}")
                continue
            
            all_job_ids.extend(rounds[rnum]['job_ids'])
            all_wandb_ids.extend(rounds[rnum]['wandb_ids'])
        
        if not all_job_ids:
            print("No job IDs found for specified rounds.")
            return
        
        self._run_cleanup(all_job_ids, all_wandb_ids, round_nums)
    
    def clean_job_ids(self, job_ids: list):
        """Clean specific job IDs."""
        self._run_cleanup(job_ids, [], [])
    
    def _run_cleanup(self, job_ids: list, wandb_ids: list, round_nums: list):
        """Run the full cleanup process."""
        print(f"\n{'=' * 50}")
        print(f"ATMOREP RUN CLEANUP")
        print(f"{'=' * 50}")
        print(f"Mode: {'DRY-RUN (preview only)' if self.dry_run else 'LIVE (files will be moved)'}")
        print(f"Job IDs to clean: {len(job_ids)}")
        print(f"Wandb IDs provided: {len(wandb_ids)}")
        
        # Create trash directory
        if round_nums:
            self.create_trash_dir(round_nums)
        else:
            self.create_trash_dir(['custom'])
        
        print(f"Trash directory: {self.trash_dir}")
        
        # Run cleanup steps
        self.clean_output_files(job_ids)
        self.clean_log_files(job_ids)
        self.clean_wandb_runs(job_ids, wandb_ids)
        self.clean_results(wandb_ids)
        self.clean_models(wandb_ids)
        
        # Print summary
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Clean up AtmoRep evaluation run files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_runs.py --round 3                    # Clean ROUND3
  python cleanup_runs.py --round 3 --dry-run          # Preview what would be cleaned
  python cleanup_runs.py --job-ids 21584933,21584934  # Clean specific jobs
  python cleanup_runs.py --round 2 --round 3          # Clean multiple rounds
  python cleanup_runs.py --list-rounds                # Show available rounds
        """
    )
    
    parser.add_argument(
        '--round', '-r',
        type=int,
        action='append',
        dest='rounds',
        help='Round number(s) to clean (can specify multiple: -r 2 -r 3)'
    )
    
    parser.add_argument(
        '--job-ids', '-j',
        type=str,
        help='Comma-separated list of SLURM job IDs to clean'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview what would be cleaned without actually moving files'
    )
    
    parser.add_argument(
        '--list-rounds', '-l',
        action='store_true',
        help='List available rounds from important_result_ids.txt'
    )
    
    parser.add_argument(
        '--base-dir', '-d',
        type=str,
        default='/work/ab1412/atmorep',
        help='Base directory for AtmoRep (default: /work/ab1412/atmorep)'
    )
    
    args = parser.parse_args()
    
    cleaner = RunCleaner(base_dir=args.base_dir, dry_run=args.dry_run)
    
    if args.list_rounds:
        cleaner.list_rounds()
        return
    
    if args.rounds:
        cleaner.clean_rounds(args.rounds)
    elif args.job_ids:
        job_ids = [jid.strip() for jid in args.job_ids.split(',')]
        cleaner.clean_job_ids(job_ids)
    else:
        parser.print_help()
        print("\nError: Must specify --round or --job-ids")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main() or 0)
