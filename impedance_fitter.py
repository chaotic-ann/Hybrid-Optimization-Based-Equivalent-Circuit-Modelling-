"""
Main Impedance Fitter Module - Updated
Integrates 5 circuit models and 8 optimization algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from circuit_models import CircuitModels
from optimization_algorithms_clean import OptimizationAlgorithms


class ImpedanceFitter:
    """
    Main class for fitting impedance data using multiple circuit models and optimization algorithms.
    """
    
    def __init__(self, model_type='piecewise', seed=1234):
        """
        Initialize the impedance fitter.
        
        Parameters:
        -----------
        model_type : str
            'piecewise', 'unified', 'rs_c', 'rs_cpe', or 'gcpe_series'
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Initialize circuit model
        self.circuit_model = CircuitModels(model_type)
        
        # Initialize optimization algorithms
        self.optimizer = OptimizationAlgorithms(
            self.circuit_model.bounds,
            self.circuit_model.lb,
            self.circuit_model.ub
        )
        
        self.model_type = model_type
        self.methods = ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE']
    
    def load_csv_data(self, filename):
        """
        Load and preprocess CSV data.
        
        Parameters:
        -----------
        filename : str
            Path to the CSV file
            
        Returns:
        --------
        tuple : (freq, mag, phase_deg, Zf)
            Processed frequency, magnitude, phase, and complex impedance data
        """
        try:
            data = pd.read_csv(filename, header=None)
            print(f"File '{filename}' loaded successfully!")
            
            freq = pd.to_numeric(data.iloc[:, 0], errors='coerce').values
            mag = pd.to_numeric(data.iloc[:, 1], errors='coerce').values
            phase_deg = pd.to_numeric(data.iloc[:, 2], errors='coerce').values
            
            valid_idx = np.isfinite(freq) & np.isfinite(mag) & np.isfinite(phase_deg)
            freq = freq[valid_idx]
            mag = mag[valid_idx]
            phase_deg = phase_deg[valid_idx]
            
            print(f"Loaded {len(freq)} valid data points.")
            
            Zf = mag * np.exp(1j * np.deg2rad(phase_deg))
            
            return freq, mag, phase_deg, Zf
            
        except FileNotFoundError:
            print(f"ERROR: File '{filename}' not found.")
            raise
        except Exception as e:
            print(f"Error loading '{filename}': {e}")
            raise
    
    def objective_function(self, p, freq_data, Z_measured):
        """
        Objective function for optimization.
        
        Parameters:
        -----------
        p : array
            Model parameters
        freq_data : tuple
            Frequency data
        Z_measured : array
            Measured impedance data
            
        Returns:
        --------
        float : Sum of squared errors
        """
        Z_model = self.circuit_model.compute_model_impedance(p, freq_data)
        return np.sum(np.abs(Z_model - Z_measured)**2)
    
    def evaluate_fit(self, p, freq_data, Z_measured, freq_full):
        """
        Evaluate fit quality metrics.
        
        Parameters:
        -----------
        p : array
            Model parameters
        freq_data : tuple
            Frequency data
        Z_measured : array
            Measured impedance data
        freq_full : array
            Full frequency array for interpolation
            
        Returns:
        --------
        tuple : (RMSE, R2, RMSE_low, R2_low, RMSE_high, R2_high, Z_fit)
        """
        if self.model_type in ['unified', 'rs_c', 'rs_cpe', 'gcpe_series']:
            # Direct comparison for non-piecewise models
            Z_fit = self.circuit_model.compute_model_impedance(p, freq_data)
            
            residual = Z_measured - Z_fit
            RMSE = np.sqrt(np.mean(np.abs(residual)**2))
            R2 = 1 - np.sum(np.abs(residual)**2) / np.sum(np.abs(Z_measured - np.mean(Z_measured))**2)
            
            # Split for reporting
            n_mid = len(Z_measured) // 2
            Z_fit_low = Z_fit[:n_mid]
            Z_fit_high = Z_fit[n_mid:]
            Z_meas_low = Z_measured[:n_mid]
            Z_meas_high = Z_measured[n_mid:]
            
            res_low = Z_meas_low - Z_fit_low
            res_high = Z_meas_high - Z_fit_high
            RMSE_low = np.sqrt(np.mean(np.abs(res_low)**2))
            R2_low = 1 - np.sum(np.abs(res_low)**2) / np.sum(np.abs(Z_meas_low - np.mean(Z_meas_low))**2)
            RMSE_high = np.sqrt(np.mean(np.abs(res_high)**2))
            R2_high = 1 - np.sum(np.abs(res_high)**2) / np.sum(np.abs(Z_meas_high - np.mean(Z_meas_high))**2)
            
            return RMSE, R2, RMSE_low, R2_low, RMSE_high, R2_high, Z_fit
            
        else:
            # Piecewise model with interpolation
            freq_low, freq_high, Z_low, Z_high = freq_data
            Z_fit = self.circuit_model.compute_model_impedance(p, freq_data)
            n_low = len(Z_low)
            Z_fit_low = Z_fit[:n_low]
            Z_fit_high = Z_fit[n_low:]
            
            freq_combined = np.concatenate([freq_low, freq_high])
            mag_fit = np.abs(Z_fit)
            phase_fit = np.angle(Z_fit)
            
            mag_interp = np.interp(freq_full, freq_combined, mag_fit)
            phase_interp = np.interp(freq_full, freq_combined, phase_fit)
            Z_fit_interp = mag_interp * np.exp(1j * phase_interp)
            
            residual = Z_measured - Z_fit_interp
            RMSE = np.sqrt(np.mean(np.abs(residual)**2))
            R2 = 1 - np.sum(np.abs(residual)**2) / np.sum(np.abs(Z_measured - np.mean(Z_measured))**2)
            
            res_low = Z_low - Z_fit_low
            res_high = Z_high - Z_fit_high
            RMSE_low = np.sqrt(np.mean(np.abs(res_low)**2))
            R2_low = 1 - np.sum(np.abs(res_low)**2) / np.sum(np.abs(Z_low - np.mean(Z_low))**2)
            RMSE_high = np.sqrt(np.mean(np.abs(res_high)**2))
            R2_high = 1 - np.sum(np.abs(res_high)**2) / np.sum(np.abs(Z_high - np.mean(Z_high))**2)
            
            return RMSE, R2, RMSE_low, R2_low, RMSE_high, R2_high, Z_fit_interp
    
    def create_results_tables(self, all_results, file_name):
        """
        Create pandas DataFrames with results.
        
        Returns:
        --------
        tuple : (metrics_df, params_df)
        """
        # Metrics table
        metrics_data = {
            "Method": self.methods,
            "Total RMSE (Ω)": [all_results[m]['RMSE'] for m in self.methods if m in all_results],
            "Total R²": [all_results[m]['R2'] for m in self.methods if m in all_results],
            "Low Freq RMSE (Ω)": [all_results[m]['RMSE_low'] for m in self.methods if m in all_results],
            "Low Freq R²": [all_results[m]['R2_low'] for m in self.methods if m in all_results],
            "High Freq RMSE (Ω)": [all_results[m]['RMSE_high'] for m in self.methods if m in all_results],
            "High Freq R²": [all_results[m]['R2_high'] for m in self.methods if m in all_results],
        }
        
        # Parameters table
        params_data = {"Method": [m for m in self.methods if m in all_results]}
        for i, param_name in enumerate(self.circuit_model.param_names):
            params_data[param_name] = [all_results[m]['params'][i] for m in self.methods if m in all_results]
        
        metrics_df = pd.DataFrame(metrics_data)
        params_df = pd.DataFrame(params_data)
        
        return metrics_df, params_df
    
    def plot_results(self, freq, mag, phase_deg, all_results, file_name, save_path=None):
        """Plot impedance fitting results with all methods."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Magnitude plot
        ax1 = axes[0, 0]
        ax1.set_title(f"Impedance Magnitude - {file_name} ({self.model_type.upper()})")
        ax1.plot(freq, mag, 'k-', linewidth=2, label='Measured', alpha=0.8)
        
        colors = ['r', 'g', 'b', 'c', 'm', 'orange', 'purple', 'brown']
        
        for i, method in enumerate(self.methods):
            if method in all_results:
                Z_fit = all_results[method]['Z_fit']
                ax1.plot(freq, np.abs(Z_fit), '--', color=colors[i % len(colors)], 
                        label=method, alpha=0.7, linewidth=1.5)
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_ylabel('|Z| (Ω)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, which="both", alpha=0.3)
        
        # Phase plot
        ax2 = axes[0, 1]
        ax2.set_title(f"Phase - {file_name} ({self.model_type.upper()})")
        ax2.plot(freq, phase_deg, 'k-', linewidth=2, label='Measured', alpha=0.8)
        
        for i, method in enumerate(self.methods):
            if method in all_results:
                Z_fit = all_results[method]['Z_fit']
                ax2.plot(freq, np.angle(Z_fit, deg=True), '--', color=colors[i % len(colors)], 
                        label=method, alpha=0.7, linewidth=1.5)
        
        ax2.set_xscale('log')
        ax2.set_ylabel('Phase (degrees)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, which="both", alpha=0.3)
        
        # RMSE comparison
        ax3 = axes[1, 0]
        ax3.set_title(f"RMSE Comparison - {file_name}")
        rmse_values = [all_results[method]['RMSE'] for method in self.methods if method in all_results]
        method_names = [method for method in self.methods if method in all_results]
        
        bars = ax3.bar(range(len(method_names)), rmse_values, color=colors[:len(method_names)])
        ax3.set_xticks(range(len(method_names)))
        ax3.set_xticklabels(method_names, rotation=45, ha='right')
        ax3.set_ylabel('RMSE (Ω)')
        ax3.set_yscale('log')
        
        # Highlight best method
        best_method = min(all_results.keys(), key=lambda k: all_results[k]['RMSE'])
        best_idx = method_names.index(best_method)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        
        # R² comparison
        ax4 = axes[1, 1]
        ax4.set_title(f"R² Comparison - {file_name}")
        r2_values = [all_results[method]['R2'] for method in method_names]
        
        bars2 = ax4.bar(range(len(method_names)), r2_values, color=colors[:len(method_names)])
        ax4.set_xticks(range(len(method_names)))
        ax4.set_xticklabels(method_names, rotation=45, ha='right')
        ax4.set_ylabel('R²')
        ax4.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='R²=0.99')
        ax4.legend()
        
        # Highlight best method
        bars2[best_idx].set_color('gold')
        bars2[best_idx].set_edgecolor('black')
        bars2[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        
        if save_path:
            plot_filename = f"{file_name}_{self.model_type}_impedance_fit.png"
            plt.savefig(os.path.join(save_path, plot_filename), dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {plot_filename}")
        
        plt.show()
    
    def process_single_file(self, filename, save_path=None):
        """
        Process a single CSV file through the complete analysis pipeline.
        
        Parameters:
        -----------
        filename : str
            Path to CSV file
        save_path : str, optional
            Directory to save results
            
        Returns:
        --------
        dict : Complete results for the file
        """
        file_name = Path(filename).stem
        print(f"\n{'='*70}")
        print(f"Processing: {file_name}")
        print(f"Model: {self.circuit_model.get_model_description()}")
        print(f"{'='*70}")
        
        # Load and preprocess data
        freq, mag, phase_deg, Zf = self.load_csv_data(filename)
        
        # Split frequency regions or prepare data
        freq_data = self.circuit_model.split_frequency_regions(freq, Zf)
        
        # Compute initial parameters
        p0 = self.circuit_model.compute_initial_params(freq, Zf, phase_deg)
        
        # Prepare data for optimization
        if self.model_type == 'piecewise':
            freq_low, freq_high, Z_low, Z_high = freq_data
            Z_all = np.concatenate([Z_low, Z_high])
        else:
            Z_all = Zf
        
        # Create objective function
        obj_fun = lambda p: self.objective_function(p, freq_data, Z_all)
        
        # Run optimization algorithms
        print("\nRunning optimization algorithms...")
        print("="*70)
        opt_results = self.optimizer.run_all_algorithms(obj_fun, p0)
        
        # Evaluate all fits
        all_results = {}
        print("\n--- Evaluating Fits ---")
        for method, params in opt_results.items():
            RMSE, R2, RMSE_low, R2_low, RMSE_high, R2_high, Z_fit = self.evaluate_fit(
                params, freq_data, Zf, freq
            )
            all_results[method] = {
                'params': params,
                'RMSE': RMSE,
                'R2': R2,
                'RMSE_low': RMSE_low,
                'R2_low': R2_low,
                'RMSE_high': RMSE_high,
                'R2_high': R2_high,
                'Z_fit': Z_fit
            }
            print(f"{method:12s}: RMSE = {RMSE:.6f} Ω, R² = {R2:.6f}")
        
        # Create results tables
        metrics_df, params_df = self.create_results_tables(all_results, file_name)
        
        # Print results
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY - {file_name} ({self.model_type.upper()})")
        print(f"{'='*70}")
        print("\n--- Fit Quality Metrics ---")
        print(metrics_df.to_string(index=False))
        print(f"\n--- Fitted Parameters ---")
        print(params_df.to_string(index=False))
        
        # Find best algorithm
        best_method = min(all_results.keys(), key=lambda k: all_results[k]['RMSE'])
        best_result = all_results[best_method]
        print(f"\n{'='*70}")
        print(f"BEST ALGORITHM: {best_method}")
        print(f"Total RMSE = {best_result['RMSE']:.6f} Ω")
        print(f"Total R² = {best_result['R2']:.6f}")
        print(f"{'='*70}")
        
        # Save results if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            metrics_df.to_csv(os.path.join(save_path, f"{file_name}_{self.model_type}_metrics.csv"), index=False)
            params_df.to_csv(os.path.join(save_path, f"{file_name}_{self.model_type}_parameters.csv"), index=False)
            print(f"\nResults saved to: {save_path}")
        
        return {
            'file_name': file_name,
            'model_type': self.model_type,
            'metrics_df': metrics_df,
            'params_df': params_df,
            'all_results': all_results,
            'best_method': best_method,
            'freq': freq,
            'mag': mag,
            'phase_deg': phase_deg
        }


# Convenience functions for easy usage
def analyze_with_model(csv_file_path, model_type, save_results=True, output_dir=None):
    """
    Analyze with specified model.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file
    model_type : str
        One of: 'piecewise', 'unified', 'rs_c', 'rs_cpe', 'gcpe_series'
    save_results : bool
        Whether to save results to files
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    dict : Analysis results
    """
    if output_dir is None:
        output_dir = f"results_{model_type}"
    
    fitter = ImpedanceFitter(model_type=model_type)
    save_path = output_dir if save_results else None
    return fitter.process_single_file(csv_file_path, save_path)


# Example usage
if __name__ == "__main__":
    print("Enhanced Modular Impedance Fitter Ready!")
    print("\nAvailable Models:")
    print("  1. piecewise: CPE || GCPE + Rs (low-f) and R + L + C (high-f)")
    print("  2. unified: CPE + GCPE for entire frequency range")
    print("  3. rs_c: Rs + C (simple model)")
    print("  4. rs_cpe: Rs + CPE")
    print("  5. gcpe_series: GCPE||CPE in series with GCPE||CPE + Rs")
    print("\nEight optimization algorithms:")
    print("  Base: SA, SLSQP")
    print("  Hybrid: DE+LGBS, DE+SLSQP, PSO+LGBS, PSO+SLSQP, DE+PSO, PSO+DE")