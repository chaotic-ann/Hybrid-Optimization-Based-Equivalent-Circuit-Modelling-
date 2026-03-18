#!/usr/bin/env python3
"""
Comprehensive Analysis Runner - Updated for 5 Models & 8 Algorithms
Automatically processes all CSV files with multiple models and generates complete reports.
"""

import os
import sys
import time
from pathlib import Path
from impedance_fitter import ImpedanceFitter
from file_manager import FileManager


class AnalysisRunner:
    """
    Automated analysis runner for batch processing impedance data.
    Updated to support all 5 models and 8 optimization algorithms.
    """
    
    def __init__(self):
        """Initialize the analysis runner."""
        self.file_manager = FileManager()
        
        # Initialize all 5 model fitters
        self.fitters = {
            'rs_c': ImpedanceFitter(model_type='rs_c'),
            'rs_cpe': ImpedanceFitter(model_type='rs_cpe'),
            'piecewise': ImpedanceFitter(model_type='piecewise'),
            'unified': ImpedanceFitter(model_type='unified'),
            'gcpe_series': ImpedanceFitter(model_type='gcpe_series')
        }
        
        self.all_results = []
    
    def run_single_file_analysis(self, csv_file_path, dirs, models_to_run=None):
        """
        Run comprehensive analysis on a single file.
        
        Parameters:
        -----------
        csv_file_path : str
            Path to CSV file
        dirs : dict
            Directory structure
        models_to_run : list, optional
            List of models to run. If None, runs all 5 models.
            Options: ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
            
        Returns:
        --------
        dict : Analysis results for all requested models
        """
        if models_to_run is None:
            models_to_run = ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
        
        file_name = Path(csv_file_path).stem
        print(f"\n{'='*70}")
        print(f"ANALYZING: {file_name}")
        print(f"{'='*70}")
        print(f"Models to run: {', '.join([m.upper() for m in models_to_run])}")
        print(f"Algorithms: SA, SLSQP, DE+LGBS, DE+SLSQP, PSO+LGBS, PSO+SLSQP, DE+PSO, PSO+DE")
        print("-" * 70)
        
        try:
            results = {}
            total_start = time.time()
            
            for model_type in models_to_run:
                print(f"\n--- Running {model_type.upper()} Model ---")
                model_start = time.time()
                
                # Run analysis
                fitter = self.fitters[model_type]
                result = fitter.process_single_file(csv_file_path, save_path=None)
                results[model_type] = result
                
                # Save individual results
                self.file_manager.save_individual_results(result, dirs, model_type)
                
                # Save plots
                self.file_manager.save_plots(
                    result['freq'], 
                    result['mag'], 
                    result['phase_deg'],
                    result['all_results'],
                    file_name,
                    model_type,
                    dirs
                )
                
                model_time = time.time() - model_start
                best_method = result['best_method']
                best_rmse = result['all_results'][best_method]['RMSE']
                best_r2 = result['all_results'][best_method]['R2']
                
                print(f"✅ {model_type.upper()} complete in {model_time:.1f}s")
                print(f"   Best: {best_method} | RMSE: {best_rmse:.6f} Ω | R²: {best_r2:.6f}")
            
            # Save comparison results if multiple models
            if len(models_to_run) > 1:
                self.file_manager.save_comparison_results(results, dirs)
            
            total_time = time.time() - total_start
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"SUMMARY - {file_name}")
            print(f"{'='*70}")
            print(f"{'Model':<15} {'Best Method':<12} {'RMSE (Ω)':<12} {'R²':<10}")
            print("-" * 70)
            
            for model_type in models_to_run:
                result = results[model_type]
                best_method = result['best_method']
                best_rmse = result['all_results'][best_method]['RMSE']
                best_r2 = result['all_results'][best_method]['R2']
                print(f"{model_type.upper():<15} {best_method:<12} {best_rmse:<12.6f} {best_r2:<10.6f}")
            
            # Find overall best
            best_overall = min(results.items(), 
                             key=lambda x: x[1]['all_results'][x[1]['best_method']]['RMSE'])
            
            print("=" * 70)
            print(f"🏆 BEST: {best_overall[0].upper()} with {best_overall[1]['best_method']}")
            print(f"⏱️  Total time: {total_time:.1f}s")
            print("=" * 70)
            
            return results
            
        except Exception as e:
            print(f"❌ ERROR analyzing {file_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_batch_analysis(self, input_directory=None, csv_files=None, 
                          analysis_name=None, models_to_run=None):
        """
        Run batch analysis on multiple files.
        
        Parameters:
        -----------
        input_directory : str, optional
            Directory containing CSV files
        csv_files : list, optional
            List of specific CSV files
        analysis_name : str, optional
            Custom name for analysis session
        models_to_run : list, optional
            List of models to run. If None, runs all 5.
            
        Returns:
        --------
        dict : Batch analysis results
        """
        # Determine files to process
        if csv_files is None:
            if input_directory is None:
                input_directory = input("Enter directory containing CSV files: ").strip()
            
            csv_files = self.file_manager.find_csv_files(input_directory)
            
            if not csv_files:
                print("❌ No CSV files found!")
                return None
        
        # Filter out any directories that might have been included
        csv_files = [f for f in csv_files if os.path.isfile(f) and f.lower().endswith('.csv')]
        
        if not csv_files:
            print("No valid CSV files found after filtering!")
            return None
        
        # Determine which models to run
        if models_to_run is None:
            print("\nSelect models to run:")
            print("1. All models (rs_c, rs_cpe, piecewise, unified, gcpe_series)")
            print("2. Simple models only (rs_c, rs_cpe)")
            print("3. Original models only (piecewise, unified)")
            print("4. Custom selection")
            
            choice = input("Enter choice (1-4, default=1): ").strip() or "1"
            
            if choice == "1":
                models_to_run = ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
            elif choice == "2":
                models_to_run = ['rs_c', 'rs_cpe']
            elif choice == "3":
                models_to_run = ['piecewise', 'unified']
            elif choice == "4":
                print("\nAvailable models: rs_c, rs_cpe, piecewise, unified, gcpe_series")
                models_input = input("Enter models (comma-separated): ").strip()
                models_to_run = [m.strip() for m in models_input.split(',')]
            else:
                models_to_run = ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
        
        # Create directory structure
        dirs = self.file_manager.create_directory_structure(analysis_name)
        
        print(f"\n{'='*70}")
        print("BATCH ANALYSIS")
        print(f"{'='*70}")
        print(f"Output directory: {dirs['main']}")
        print(f"Files to process: {len(csv_files)}")
        print(f"Models to run: {', '.join([m.upper() for m in models_to_run])}")
        print(f"Algorithms: 8 (DE, PSO, SA, SLSQP + 4 hybrids)")
        print("=" * 70)
        
        # Process each file
        total_start_time = time.time()
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n[{i}/{len(csv_files)}] Processing: {os.path.basename(csv_file)}")
            
            result = self.run_single_file_analysis(csv_file, dirs, models_to_run)
            if result:
                self.all_results.append(result)
        
        total_time = time.time() - total_start_time
        
        # Generate batch summary
        print(f"\n{'='*70}")
        print("GENERATING BATCH SUMMARY...")
        print(f"{'='*70}")
        self.file_manager.save_batch_summary(self.all_results, dirs)
        
        # Create final report
        self.file_manager.create_final_report(dirs, self.all_results)
        
        # Print final summary
        print(f"\n{'='*70}")
        print("BATCH ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        print(f"✅ Successfully processed: {len(self.all_results)}/{len(csv_files)} files")
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"📁 Results saved to: {dirs['main']}")
        print(f"📊 Models analyzed: {', '.join([m.upper() for m in models_to_run])}")
        print("=" * 70)
        
        # Print best methods statistics
        self.print_method_statistics(models_to_run)
        
        return {
            'results': self.all_results,
            'directories': dirs,
            'processing_time': total_time,
            'success_rate': len(self.all_results) / len(csv_files) * 100,
            'models_used': models_to_run
        }
    
    def print_method_statistics(self, models_to_run):
        """Print statistics about which methods performed best."""
        print(f"\n{'='*70}")
        print("METHOD PERFORMANCE STATISTICS")
        print(f"{'='*70}")
        
        methods = ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE']
        
        for model in models_to_run:
            method_wins = {m: 0 for m in methods}
            total_count = 0
            
            for result in self.all_results:
                if model in result:
                    best_method = result[model]['best_method']
                    if best_method in method_wins:
                        method_wins[best_method] += 1
                    total_count += 1
            
            if total_count > 0:
                print(f"\n{model.upper()} Model:")
                sorted_methods = sorted(method_wins.items(), key=lambda x: x[1], reverse=True)
                for method, wins in sorted_methods[:3]:  # Top 3
                    win_rate = (wins / total_count) * 100
                    if wins > 0:
                        print(f"  {method:<12} - {wins:2d} wins ({win_rate:5.1f}%)")
    
    def run_directory_analysis(self, directory_path, models_to_run=None):
        """
        Automatically find and process all CSV files in a directory.
        
        Parameters:
        -----------
        directory_path : str
            Path to directory containing CSV files
        models_to_run : list, optional
            List of models to run
        """
        if not os.path.exists(directory_path):
            print(f"❌ ERROR: Directory not found: {directory_path}")
            return None
        
        analysis_name = f"batch_{Path(directory_path).name}_{self.file_manager.timestamp}"
        return self.run_batch_analysis(input_directory=directory_path, 
                                      analysis_name=analysis_name,
                                      models_to_run=models_to_run)
    
    def run_interactive_analysis(self):
        """Run analysis in interactive mode."""
        print("🔬 INTERACTIVE IMPEDANCE ANALYSIS")
        print("=" * 70)
        print("1. Analyze single file (all 5 models)")
        print("2. Analyze single file (select models)")
        print("3. Analyze directory of CSV files")
        print("4. Analyze specific list of files")
        
        choice = input("\nSelect option (1/2/3/4): ").strip()
        
        if choice == "1":
            csv_file = input("Enter CSV file path: ").strip()
            if not os.path.exists(csv_file):
                print(f"❌ ERROR: File not found: {csv_file}")
                return
            
            dirs = self.file_manager.create_directory_structure()
            models_to_run = ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
            result = self.run_single_file_analysis(csv_file, dirs, models_to_run)
            if result:
                self.file_manager.create_final_report(dirs, [result])
        
        elif choice == "2":
            csv_file = input("Enter CSV file path: ").strip()
            if not os.path.exists(csv_file):
                print(f"❌ ERROR: File not found: {csv_file}")
                return
            
            print("\nSelect models:")
            print("Available: rs_c, rs_cpe, piecewise, unified, gcpe_series")
            models_input = input("Enter models (comma-separated): ").strip()
            models_to_run = [m.strip() for m in models_input.split(',')]
            
            dirs = self.file_manager.create_directory_structure()
            result = self.run_single_file_analysis(csv_file, dirs, models_to_run)
            if result:
                self.file_manager.create_final_report(dirs, [result])
        
        elif choice == "3":
            directory = input("Enter directory path: ").strip()
            self.run_directory_analysis(directory)
        
        elif choice == "4":
            print("Enter CSV file paths (one per line, empty line to finish):")
            csv_files = []
            while True:
                file_path = input().strip()
                if not file_path:
                    break
                if os.path.exists(file_path):
                    csv_files.append(file_path)
                else:
                    print(f"⚠️  WARNING: File not found: {file_path}")
            
            if csv_files:
                self.run_batch_analysis(csv_files=csv_files)
            else:
                print("❌ ERROR: No valid files provided!")
        
        else:
            print("❌ ERROR: Invalid choice!")


def main():
    """Main function for running analysis."""
    runner = AnalysisRunner()
    
    print("🔬 COMPREHENSIVE IMPEDANCE SPECTROSCOPY ANALYZER")
    print("=" * 70)
    print("5 Circuit Models:")
    print("   - Rs+C (2 params)")
    print("   - Rs+CPE (5 params)")
    print("   - Piecewise (8 params)")
    print("   - Unified (9 params)")
    print("   - GCPE Series (17 params)")
    print("\n8 Optimization Algorithms:")
    print("   - Base: SA, SLSQP")
    print("   - Hybrid: DE+LGBS, DE+SLSQP, PSO+LGBS, PSO+SLSQP, DE+PSO, PSO+DE")
    print("\nComplete Output Generation:")
    print("   - Individual results, comparisons, plots, summaries")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # Command line mode
        input_path = sys.argv[1]
        
        # Optional: specify models from command line
        models_to_run = None
        if len(sys.argv) > 2:
            models_arg = sys.argv[2]
            if models_arg in ['simple', 'original', 'all']:
                if models_arg == 'simple':
                    models_to_run = ['rs_c', 'rs_cpe']
                elif models_arg == 'original':
                    models_to_run = ['piecewise', 'unified']
                else:
                    models_to_run = ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
            else:
                models_to_run = [m.strip() for m in models_arg.split(',')]
        
        if os.path.isfile(input_path) and input_path.endswith('.csv'):
            # Single file
            print(f"\n📄 Analyzing single file: {input_path}")
            if models_to_run is None:
                models_to_run = ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
            dirs = runner.file_manager.create_directory_structure()
            result = runner.run_single_file_analysis(input_path, dirs, models_to_run)
            if result:
                runner.file_manager.create_final_report(dirs, [result])
        
        elif os.path.isdir(input_path):
            # Directory
            print(f"\n📁 Analyzing directory: {input_path}")
            runner.run_directory_analysis(input_path, models_to_run)
        
        else:
            print(f"❌ ERROR: Invalid path: {input_path}")
    
    else:
        # Interactive mode
        runner.run_interactive_analysis()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()