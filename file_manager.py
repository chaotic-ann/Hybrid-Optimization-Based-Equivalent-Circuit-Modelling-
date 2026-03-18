"""
File Manager Module - Updated for 5 Models & 8 Algorithms
Handles all file I/O operations including CSV exports, PNG plots, and batch processing.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import json
from datetime import datetime


class FileManager:
    """
    Manages all file operations for impedance fitting analysis.
    """
    
    def __init__(self, base_output_dir="results"):
        """
        Initialize file manager.
        
        Parameters:
        -----------
        base_output_dir : str
            Base directory for all outputs
        """
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def create_directory_structure(self, analysis_name=None):
        """
        Create organized directory structure for outputs.
        
        Parameters:
        -----------
        analysis_name : str, optional
            Custom name for this analysis session
            
        Returns:
        --------
        dict : Directory paths
        """
        if analysis_name is None:
            analysis_name = f"analysis_{self.timestamp}"
        
        # Main analysis directory
        main_dir = os.path.join(self.base_output_dir, analysis_name)
        
        # Create subdirectories for all 5 models
        dirs = {
            'main': main_dir,
            'rs_c': os.path.join(main_dir, 'rs_c_model'),
            'rs_cpe': os.path.join(main_dir, 'rs_cpe_model'),
            'piecewise': os.path.join(main_dir, 'piecewise_model'),
            'unified': os.path.join(main_dir, 'unified_model'),
            'gcpe_series': os.path.join(main_dir, 'gcpe_series_model'),
            'comparison': os.path.join(main_dir, 'model_comparison'),
            'plots': os.path.join(main_dir, 'plots'),
            'csvs': os.path.join(main_dir, 'csv_exports'),
            'summary': os.path.join(main_dir, 'summary_reports'),
            'raw_data': os.path.join(main_dir, 'processed_data')
        }
        
        # Create all directories
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return dirs
    
    def save_individual_results(self, result, dirs, model_type):
        """
        Save individual file analysis results as Excel files.
        
        Parameters:
        -----------
        result : dict
            Analysis result from impedance fitter
        dirs : dict
            Directory structure
        model_type : str
            'rs_c', 'rs_cpe', 'piecewise', 'unified', or 'gcpe_series'
        """
        file_name = result['file_name']
        model_dir = dirs[model_type]
        
        # Save as Excel with multiple sheets
        excel_filename = os.path.join(model_dir, f"{file_name}_results.xlsx")
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Sheet 1: Metrics
            result['metrics_df'].to_excel(writer, sheet_name='Metrics', index=False)
            
            # Sheet 2: Parameters
            result['params_df'].to_excel(writer, sheet_name='Parameters', index=False)
            
            # Sheet 3: Fitted Data (all methods)
            self._create_fitted_data_df(
                result['freq'], 
                result['mag'], 
                result['phase_deg'], 
                result['all_results']
            ).to_excel(writer, sheet_name='Fitted_Data', index=False)
            
            # Sheet 4: Best Method Only
            best_method = result['best_method']
            Z_best = result['all_results'][best_method]['Z_fit']
            best_df = pd.DataFrame({
                'Frequency_Hz': result['freq'],
                'Magnitude_Measured_Ohm': result['mag'],
                'Phase_Measured_Deg': result['phase_deg'],
                'Magnitude_Fit_Ohm': np.abs(Z_best),
                'Phase_Fit_Deg': np.angle(Z_best, deg=True),
                'Real_Fit_Ohm': np.real(Z_best),
                'Imag_Fit_Ohm': np.imag(Z_best),
                'Best_Method': [best_method] * len(result['freq'])
            })
            best_df.to_excel(writer, sheet_name='Best_Fit', index=False)
            
            # Sheet 5: Residuals
            self._create_residuals_df(
                result['freq'],
                result['mag'],
                result['phase_deg'],
                result['all_results']
            ).to_excel(writer, sheet_name='Residuals', index=False)
        
        print(f"OK: {model_type.upper()} results saved as Excel: {file_name}_results.xlsx")
        
        # Also save individual CSV files for backward compatibility (optional)
        result['metrics_df'].to_csv(
            os.path.join(model_dir, f"{file_name}_metrics.csv"), 
            index=False
        )
        result['params_df'].to_csv(
            os.path.join(model_dir, f"{file_name}_parameters.csv"), 
            index=False
        )
    
    def _create_fitted_data_df(self, freq, mag_measured, phase_measured, all_results):
        """Create fitted data DataFrame."""
        methods = ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE']
        
        csv_data = {
            'Frequency_Hz': freq,
            'Magnitude_Measured_Ohm': mag_measured,
            'Phase_Measured_Deg': phase_measured
        }
        
        for method in methods:
            if method in all_results:
                Z_fit = all_results[method]['Z_fit']
                safe_method = method.replace('+', '_').replace('-', '_')
                
                csv_data[f'Magnitude_{safe_method}_Ohm'] = np.abs(Z_fit)
                csv_data[f'Phase_{safe_method}_Deg'] = np.angle(Z_fit, deg=True)
                csv_data[f'Real_{safe_method}_Ohm'] = np.real(Z_fit)
                csv_data[f'Imag_{safe_method}_Ohm'] = np.imag(Z_fit)
        
        return pd.DataFrame(csv_data)
    
    def _create_residuals_df(self, freq, mag_measured, phase_measured, all_results):
        """Create residuals DataFrame."""
        methods = ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE']
        
        residuals_data = {'Frequency_Hz': freq}
        Z_measured = mag_measured * np.exp(1j * np.deg2rad(phase_measured))
        
        for method in methods:
            if method in all_results:
                Z_fit = all_results[method]['Z_fit']
                residual = Z_measured - Z_fit
                safe_method = method.replace('+', '_').replace('-', '_')
                
                residuals_data[f'Residual_Mag_{safe_method}_Ohm'] = np.abs(residual)
                residuals_data[f'Residual_Phase_{safe_method}_Deg'] = np.angle(residual, deg=True)
                residuals_data[f'Residual_Real_{safe_method}_Ohm'] = np.real(residual)
                residuals_data[f'Residual_Imag_{safe_method}_Ohm'] = np.imag(residual)
        
        return pd.DataFrame(residuals_data)
    
    def save_fitted_data_csv(self, freq, mag_measured, phase_measured, all_results, file_name, save_dir, model_type):
        """
        Save fitted data as CSV files for Origin plotting.
        Now supports all 8 optimization methods.
        """
        methods = ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE']
        
        # 1. Complete fitted data CSV
        csv_data = {
            'Frequency_Hz': freq,
            'Magnitude_Measured_Ohm': mag_measured,
            'Phase_Measured_Deg': phase_measured
        }
        
        # Add fitted data for each method
        for method in methods:
            if method in all_results:
                Z_fit = all_results[method]['Z_fit']
                mag_fit = np.abs(Z_fit)
                phase_fit = np.angle(Z_fit, deg=True)
                
                # Sanitize method name for column headers (replace + with _)
                safe_method = method.replace('+', '_')
                
                csv_data[f'Magnitude_{safe_method}_Ohm'] = mag_fit
                csv_data[f'Phase_{safe_method}_Deg'] = phase_fit
                csv_data[f'Real_{safe_method}_Ohm'] = np.real(Z_fit)
                csv_data[f'Imag_{safe_method}_Ohm'] = np.imag(Z_fit)
        
        # Save complete data
        df = pd.DataFrame(csv_data)
        csv_filename = os.path.join(save_dir, f"{file_name}_{model_type}_all_methods.csv")
        df.to_csv(csv_filename, index=False)
        
        # 2. Best method CSV
        best_method = min(all_results.keys(), key=lambda k: all_results[k]['RMSE'])
        Z_best = all_results[best_method]['Z_fit']
        
        best_data = {
            'Frequency_Hz': freq,
            'Magnitude_Measured_Ohm': mag_measured,
            'Phase_Measured_Deg': phase_measured,
            'Magnitude_Best_Fit_Ohm': np.abs(Z_best),
            'Phase_Best_Fit_Deg': np.angle(Z_best, deg=True),
            'Real_Best_Fit_Ohm': np.real(Z_best),
            'Imag_Best_Fit_Ohm': np.imag(Z_best),
            'Best_Method': [best_method] * len(freq),
            'Model_Type': [model_type] * len(freq)
        }
        
        df_best = pd.DataFrame(best_data)
        csv_best_filename = os.path.join(save_dir, f"{file_name}_{model_type}_best_method.csv")
        df_best.to_csv(csv_best_filename, index=False)
        
        # 3. Residuals CSV
        residuals_data = {'Frequency_Hz': freq}
        
        for method in methods:
            if method in all_results:
                Z_fit = all_results[method]['Z_fit']
                Z_measured = mag_measured * np.exp(1j * np.deg2rad(phase_measured))
                residual = Z_measured - Z_fit
                
                safe_method = method.replace('+', '_')
                
                residuals_data[f'Residual_Magnitude_{safe_method}_Ohm'] = np.abs(residual)
                residuals_data[f'Residual_Phase_{safe_method}_Deg'] = np.angle(residual, deg=True)
                residuals_data[f'Residual_Real_{safe_method}_Ohm'] = np.real(residual)
                residuals_data[f'Residual_Imag_{safe_method}_Ohm'] = np.imag(residual)
        
        df_residuals = pd.DataFrame(residuals_data)
        csv_residuals_filename = os.path.join(save_dir, f"{file_name}_{model_type}_residuals.csv")
        df_residuals.to_csv(csv_residuals_filename, index=False)
        
        print(f"  CSV files saved: all_methods, best_method, residuals")
    
    def save_comparison_results(self, comparison_results, dirs):
        """
        Save model comparison results.
        Updated to support all 5 models and 8 algorithms.
        
        Parameters:
        -----------
        comparison_results : dict
            Results from multiple models
        dirs : dict
            Directory structure
        """
        comparison_dir = dirs['comparison']
        
        # Model comparison summary
        summary_data = []
        for model_type, result in comparison_results.items():
            best_method = result['best_method']
            best_result = result['all_results'][best_method]
            
            summary_data.append({
                'Model_Type': model_type.upper(),
                'Best_Method': best_method,
                'Total_RMSE_Ohm': best_result['RMSE'],
                'Total_R2': best_result['R2'],
                'Low_Freq_RMSE_Ohm': best_result['RMSE_low'],
                'Low_Freq_R2': best_result['R2_low'],
                'High_Freq_RMSE_Ohm': best_result['RMSE_high'],
                'High_Freq_R2': best_result['R2_high'],
                'Parameters_Count': len(best_result['params']),
                'File_Name': result['file_name']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(comparison_dir, "model_comparison_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Detailed comparison CSV - all 8 methods
        detailed_data = []
        methods = ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE']
        
        for model_type, result in comparison_results.items():
            for method in methods:
                if method in result['all_results']:
                    method_result = result['all_results'][method]
                    detailed_data.append({
                        'Model_Type': model_type.upper(),
                        'Method': method,
                        'RMSE': method_result['RMSE'],
                        'R2': method_result['R2'],
                        'RMSE_Low': method_result['RMSE_low'],
                        'R2_Low': method_result['R2_low'],
                        'RMSE_High': method_result['RMSE_high'],
                        'R2_High': method_result['R2_high'],
                        'Best_Method': '✓' if method == result['best_method'] else '',
                        'File_Name': result['file_name']
                    })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_file = os.path.join(comparison_dir, "detailed_method_comparison.csv")
        detailed_df.to_csv(detailed_file, index=False)
        
        print(f"Comparison results saved: summary and detailed analysis")
    
    def save_plots(self, freq, mag, phase_deg, all_results, file_name, model_type, dirs):
        """
        Save impedance fitting plots.
        Updated to support 8 optimization methods.
        
        Parameters:
        -----------
        freq : array
            Frequency data
        mag : array
            Magnitude data
        phase_deg : array
            Phase data
        all_results : dict
            Results from all methods
        file_name : str
            File name
        model_type : str
            Model type
        dirs : dict
            Directory structure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Define colors for 8 methods
        colors = ['r', 'g', 'b', 'c', 'm', 'orange', 'purple', 'brown']
        methods = ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE']
        
        # Main magnitude comparison plot
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_title(f"Magnitude Fit - {file_name} ({model_type.upper()})", fontsize=12, fontweight='bold')
        ax1.plot(freq, mag, 'k-', linewidth=2.5, label='Measured', alpha=0.8)
        
        for i, method in enumerate(methods):
            if method in all_results:
                Z_fit = all_results[method]['Z_fit']
                ax1.plot(freq, np.abs(Z_fit), '--', color=colors[i], 
                        label=method, alpha=0.7, linewidth=1.5)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylabel('|Z| (Ω)', fontsize=11)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, which="both", alpha=0.3)
        
        # Phase comparison plot
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_title(f"Phase Fit - {file_name} ({model_type.upper()})", fontsize=12, fontweight='bold')
        ax2.plot(freq, phase_deg, 'k-', linewidth=2.5, label='Measured', alpha=0.8)
        
        for i, method in enumerate(methods):
            if method in all_results:
                Z_fit = all_results[method]['Z_fit']
                ax2.plot(freq, np.angle(Z_fit, deg=True), '--', color=colors[i], 
                        label=method, alpha=0.7, linewidth=1.5)
        
        ax2.set_xscale('log')
        ax2.set_ylabel('Phase (degrees)', fontsize=11)
        ax2.set_xlabel('Frequency (Hz)', fontsize=11)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, which="both", alpha=0.3)
        
        # Residuals plot
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_title(f"Magnitude Residuals - {file_name}", fontsize=12, fontweight='bold')
        Z_measured = mag * np.exp(1j * np.deg2rad(phase_deg))
        
        for i, method in enumerate(methods):
            if method in all_results:
                Z_fit = all_results[method]['Z_fit']
                residual = Z_measured - Z_fit
                ax3.plot(freq, np.abs(residual), color=colors[i], 
                        label=method, alpha=0.7, linewidth=1.5)
        
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_ylabel('|Residual| (Ω)', fontsize=11)
        ax3.set_xlabel('Frequency (Hz)', fontsize=11)
        ax3.legend(fontsize=8)
        ax3.grid(True, which="both", alpha=0.3)
        
        # RMSE comparison bar chart
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_title(f"RMSE Comparison - {file_name}", fontsize=12, fontweight='bold')
        
        rmse_values = []
        method_names = []
        for method in methods:
            if method in all_results:
                rmse_values.append(all_results[method]['RMSE'])
                method_names.append(method)
        
        bars = ax4.bar(range(len(method_names)), rmse_values, 
                      color=colors[:len(method_names)])
        ax4.set_xticks(range(len(method_names)))
        ax4.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('RMSE (Ω)', fontsize=11)
        ax4.set_yscale('log')
        
        # Highlight best method
        best_method = min(all_results.keys(), key=lambda k: all_results[k]['RMSE'])
        best_idx = method_names.index(best_method)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(dirs['plots'], f"{file_name}_{model_type}_analysis.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Plot saved: {file_name}_{model_type}_analysis.png")
    
    def save_batch_summary(self, all_results, dirs):
        """
        Save summary of batch processing results.
        Updated for all 5 models.
        
        Parameters:
        -----------
        all_results : list
            List of all analysis results
        dirs : dict
            Directory structure
        """
        if not all_results:
            return
        
        summary_dir = dirs['summary']
        
        # Overall batch summary
        batch_data = []
        models = ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
        
        for result in all_results:
            for model in models:
                if model in result:
                    model_result = result[model]
                    batch_data.append({
                        'File_Name': model_result['file_name'],
                        'Model': model.upper(),
                        'Best_Method': model_result['best_method'],
                        'Best_RMSE': model_result['all_results'][model_result['best_method']]['RMSE'],
                        'Best_R2': model_result['all_results'][model_result['best_method']]['R2'],
                        'Parameters_Count': len(model_result['all_results'][model_result['best_method']]['params'])
                    })
        
        batch_df = pd.DataFrame(batch_data)
        batch_file = os.path.join(summary_dir, "batch_analysis_summary.csv")
        batch_df.to_csv(batch_file, index=False)
        
        # Method performance statistics
        method_stats = self.calculate_method_statistics(all_results)
        method_stats_file = os.path.join(summary_dir, "method_performance_statistics.csv")
        method_stats.to_csv(method_stats_file, index=False)
        
        # Save analysis configuration
        config = {
            'analysis_timestamp': self.timestamp,
            'total_files_processed': len(all_results),
            'models_used': models,
            'optimization_methods': ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE'],
            'output_directory': dirs['main']
        }
        
        config_file = os.path.join(summary_dir, "analysis_configuration.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Batch summary saved with {len(all_results)} files analyzed")
    
    def calculate_method_statistics(self, all_results):
        """
        Calculate performance statistics for all methods.
        Updated for 8 methods and 5 models.
        """
        methods = ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE']
        models = ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
        
        stats_data = []
        
        for model in models:
            for method in methods:
                rmse_values = []
                r2_values = []
                wins = 0
                total_runs = 0
                
                for result in all_results:
                    if model in result and method in result[model]['all_results']:
                        method_result = result[model]['all_results'][method]
                        rmse_values.append(method_result['RMSE'])
                        r2_values.append(method_result['R2'])
                        total_runs += 1
                        
                        if method == result[model]['best_method']:
                            wins += 1
                
                if rmse_values:
                    stats_data.append({
                        'Model': model.upper(),
                        'Method': method,
                        'Total_Runs': total_runs,
                        'Wins': wins,
                        'Win_Rate_%': (wins / total_runs) * 100,
                        'Avg_RMSE': np.mean(rmse_values),
                        'Std_RMSE': np.std(rmse_values),
                        'Min_RMSE': np.min(rmse_values),
                        'Max_RMSE': np.max(rmse_values),
                        'Avg_R2': np.mean(r2_values),
                        'Std_R2': np.std(r2_values)
                    })
        
        return pd.DataFrame(stats_data)
    
    def find_csv_files(self, directory, pattern="*.csv"):
        """
        Find all CSV files in a directory.
        
        Parameters:
        -----------
        directory : str
            Directory to search
        pattern : str
            File pattern to match
            
        Returns:
        --------
        list : List of CSV file paths
        """
        search_pattern = os.path.join(directory, pattern)
        all_matches = glob.glob(search_pattern)
        
        # Filter to only include actual files (not directories)
        csv_files = [f for f in all_matches if os.path.isfile(f)]
        
        if csv_files:
            print(f"Found {len(csv_files)} CSV files in {directory}")
            for i, file in enumerate(csv_files, 1):
                print(f"  {i}. {os.path.basename(file)}")
        else:
            print(f"No CSV files found in {directory}")
        
        return csv_files
    
    def create_final_report(self, dirs, all_results):
        """
        Create a comprehensive final report.
        Updated for 5 models and 8 algorithms.
        """
        report_file = os.path.join(dirs['summary'], "FINAL_ANALYSIS_REPORT.txt")
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ENHANCED IMPEDANCE SPECTROSCOPY ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Files Processed: {len(all_results)}\n")
            f.write(f"Output Directory: {dirs['main']}\n\n")
            
            f.write("MODELS ANALYZED (5 Total):\n")
            f.write("1. Rs + C: Simple series resistance and capacitance (2 params)\n")
            f.write("2. Rs + CPE: Series resistance with constant phase element (5 params)\n")
            f.write("3. Piecewise: CPE || GCPE + Rs (low-f) and R + L + C (high-f) (8 params)\n")
            f.write("4. Unified: CPE + GCPE for entire frequency range (9 params)\n")
            f.write("5. GCPE Series: Two GCPE||CPE blocks in series with Rs (17 params)\n\n")
            
            f.write("OPTIMIZATION ALGORITHMS (8 Total):\n")
            f.write("Base Algorithms:\n")
            f.write("  - SA: Simulated Annealing (global)\n")
            f.write("  - SLSQP: Sequential Least Squares Programming (local)\n")
            f.write("Hybrid Algorithms:\n")
            f.write("  - DE+LGBS: Differential Evolution + L-BFGS-B\n")
            f.write("  - DE+SLSQP: Differential Evolution + SLSQP\n")
            f.write("  - PSO+LGBS: Particle Swarm + L-BFGS-B\n")
            f.write("  - PSO+SLSQP: Particle Swarm + SLSQP\n")
            f.write("  - DE+PSO: Differential Evolution + Particle Swarm (DE first)\n")
            f.write("  - PSO+DE: Particle Swarm + Differential Evolution (PSO first)\n\n")
            
            f.write("OUTPUT FILES GENERATED:\n")
            f.write(" Individual Results (per model):\n")
            f.write("   - *_metrics.csv: Fit quality metrics for all 8 algorithms\n")
            f.write("   - *_parameters.csv: Fitted parameters for all 8 algorithms\n")
            f.write("   - *_all_methods.csv: All methods fitted data\n")
            f.write("   - *_best_method.csv: Best method only\n")
            f.write("   - *_residuals.csv: Residual analysis\n")
            f.write("   - *_analysis.png: Comprehensive plots\n\n")
            
            f.write(" Summary Reports:\n")
            f.write("   - batch_analysis_summary.csv: All files, all models\n")
            f.write("   - method_performance_statistics.csv: Algorithm statistics\n")
            f.write("   - model_comparison_summary.csv: Best results per model\n")
            f.write("   - detailed_method_comparison.csv: Complete method breakdown\n")
            f.write("   - analysis_configuration.json: Analysis settings\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("ANALYSIS COMPLETE - ALL FILES SAVED\n")
            f.write("=" * 80 + "\n")
        
        print(f"Final report saved: FINAL_ANALYSIS_REPORT.txt")