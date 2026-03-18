#!/usr/bin/env python3
"""
Main script for running impedance spectroscopy analysis with 5 models and 8 algorithms.
"""

import os
import sys
from impedance_fitter import analyze_with_model, ImpedanceFitter


def main():
    """Main function to run impedance analysis."""
    
    print("🔬 Enhanced Impedance Spectroscopy Analyzer")
    print("=" * 60)
    print("5 Circuit Models | 8 Optimization Algorithms")
    print("=" * 60)
    
    # Check if CSV file is provided as command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else 'unified'
        
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found!")
            return
        
        print(f"\nAnalyzing file: {csv_file}")
        print(f"Model: {model_type}")
        result = analyze_with_model(csv_file, model_type)
        print(f"✅ Analysis complete! Best method: {result['best_method']}")
        return
    
    # Interactive mode
    print("\nAvailable circuit models:")
    print("1. Rs + C (Simple capacitive)")
    print("2. Rs + CPE (Non-ideal capacitive)")
    print("3. Piecewise (CPE||GCPE + Rs / RLC)")
    print("4. Unified (CPE + GCPE full range)")
    print("5. GCPE Series (Two GCPE||CPE blocks)")
    print("6. Compare ALL models")
    
    choice = input("\nSelect model (1-6): ").strip()
    csv_file = input("Enter path to CSV file: ").strip()
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return
    
    model_map = {
        '1': 'rs_c',
        '2': 'rs_cpe',
        '3': 'piecewise',
        '4': 'unified',
        '5': 'gcpe_series'
    }
    
    if choice in model_map:
        model_type = model_map[choice]
        print(f"\n🔧 Running {model_type.upper()} model analysis...")
        result = analyze_with_model(csv_file, model_type)
        print(f"\n✅ Analysis complete!")
        print(f"Best method: {result['best_method']}")
        print(f"Best RMSE: {result['all_results'][result['best_method']]['RMSE']:.6f} Ω")
        print(f"Best R²: {result['all_results'][result['best_method']]['R2']:.6f}")
        
    elif choice == '6':
        print("\n🔧 Running ALL models for comparison...")
        print("=" * 60)
        
        models = ['rs_c', 'rs_cpe', 'piecewise', 'unified', 'gcpe_series']
        results = {}
        
        for model in models:
            print(f"\n--- Analyzing with {model.upper()} model ---")
            try:
                result = analyze_with_model(csv_file, model, save_results=True, 
                                          output_dir=f"results_comparison/{model}")
                results[model] = result
                print(f"✅ {model.upper()}: Best = {result['best_method']}, "
                      f"RMSE = {result['all_results'][result['best_method']]['RMSE']:.6f}")
            except Exception as e:
                print(f"❌ {model.upper()}: Error - {str(e)}")
        
        # Print comparison summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<15} {'Best Method':<12} {'RMSE (Ω)':<12} {'R²':<10}")
        print("-" * 60)
        
        for model, result in results.items():
            best_method = result['best_method']
            best_rmse = result['all_results'][best_method]['RMSE']
            best_r2 = result['all_results'][best_method]['R2']
            print(f"{model.upper():<15} {best_method:<12} {best_rmse:<12.6f} {best_r2:<10.6f}")
        
        # Find overall best
        best_overall = min(results.items(), 
                          key=lambda x: x[1]['all_results'][x[1]['best_method']]['RMSE'])
        print("=" * 60)
        print(f"🏆 BEST OVERALL: {best_overall[0].upper()} with {best_overall[1]['best_method']}")
        print("=" * 60)
        
    else:
        print("Invalid choice!")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()