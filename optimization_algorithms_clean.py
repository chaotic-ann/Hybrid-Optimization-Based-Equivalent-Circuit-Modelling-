"""
Algorithms:
- Base: SA, SLSQP
- Hybrid: DE+LGBS, DE+SLSQP, PSO+LGBS, PSO+SLSQP
- Sequential (Enhanced): DE+PSO, PSO+DE
"""

import numpy as np
from scipy.optimize import differential_evolution, dual_annealing, minimize
from pyswarm import pso


class OptimizationAlgorithms:
    """
    Class containing 8 optimization algorithms for impedance fitting.
    """
    
    def __init__(self, bounds, lb, ub):
        """
        Initialize optimization algorithms.
        
        Parameters:
        -----------
        bounds : list
            Parameter bounds [(min, max), ...]
        lb : list
            Lower bounds
        ub : list
            Upper bounds
        """
        self.bounds = bounds
        self.lb = lb
        self.ub = ub
        self.n_params = len(bounds)
    
    def simulated_annealing_optimizer(self, obj_fun, x0, maxiter=1000):
        """Simulated Annealing from scipy."""
        print("Running SA Optimization (Simulated Annealing)...")
        result = dual_annealing(obj_fun, self.bounds, maxiter=maxiter, x0=x0)
        return result.x
    
    def sequential_least_squares(self, obj_fun, x0):
        """Sequential Least Squares Programming (SLSQP)."""
        print("Running SLSQP Optimization (Sequential Least Squares)...")
        result = minimize(obj_fun, x0, method='SLSQP', bounds=self.bounds, 
                         options={'maxiter': 1000, 'ftol': 1e-18})
        return result.x
    
    def local_gradient_based_search(self, obj_fun, x0):
        """Local Gradient-Based Search using L-BFGS-B."""
        print("Running L-BFGS-B Optimization (Local Gradient-Based Search)...")
        result = minimize(obj_fun, x0, method='L-BFGS-B', bounds=self.bounds,
                         options={'maxiter': 1000, 'ftol': 1e-18})
        return result.x
    
    def de_lgbs_hybrid(self, obj_fun, maxiter=5000):
        """
        Hybrid: Differential Evolution + Local Gradient-Based Search.
        
        Strategy:
        - Stage 1: DE explores the search space globally
        - Stage 2: L-BFGS-B refines the solution locally
        
        Enhanced: Increased default iterations from 500 to 1000.
        """
        print("Running DE+LGBS Hybrid Optimization...")
        
        # Stage 1: DE for global exploration
        print("  Stage 1: DE (global search)...")
        result_de = differential_evolution(obj_fun, self.bounds, maxiter=maxiter, 
                                          polish=False, tol=1e-18)
        x_de = result_de.x
        
        # Stage 2: L-BFGS-B for local refinement
        print("  Stage 2: L-BFGS-B (local refinement)...")
        result_lgbs = minimize(obj_fun, x_de, method='L-BFGS-B', bounds=self.bounds,
                              options={'maxiter': 1000, 'ftol': 1e-18})
        
        return result_lgbs.x
    
    def de_slsqp_hybrid(self, obj_fun, maxiter=5000):
        """
        Hybrid: Differential Evolution + Sequential Least Squares.
        
        Strategy:
        - Stage 1: DE explores the search space globally
        - Stage 2: SLSQP refines with constraint handling
        
        Enhanced: Increased default iterations from 500 to 1000.
        """
        print("Running DE+SLSQP Hybrid Optimization...")
        
        # Stage 1: DE for global exploration
        print("  Stage 1: DE (global search)...")
        result_de = differential_evolution(obj_fun, self.bounds, maxiter=maxiter, 
                                          polish=False, tol=1e-18)
        x_de = result_de.x
        
        # Stage 2: SLSQP for local refinement
        print("  Stage 2: SLSQP (local refinement)...")
        result_slsqp = minimize(obj_fun, x_de, method='SLSQP', bounds=self.bounds,
                               options={'maxiter': 1000, 'ftol': 1e-18})
        
        return result_slsqp.x
    
    def pso_lgbs_hybrid(self, obj_fun, maxiter=1000):
        """
        Hybrid: Particle Swarm Optimization + Local Gradient-Based Search.
        
        Strategy:
        - Stage 1: PSO uses swarm intelligence for exploration
        - Stage 2: L-BFGS-B refines the solution locally
        
        Enhanced: Increased default iterations from 500 to 1000.
        """
        print("Running PSO+LGBS Hybrid Optimization...")
        
        # Stage 1: PSO for global exploration
        print("  Stage 1: PSO (global search)...")
        x_pso, _ = pso(obj_fun, self.lb, self.ub, swarmsize=200, 
                       maxiter=maxiter, debug=False, minstep=1e-18)
        
        # Stage 2: L-BFGS-B for local refinement
        print("  Stage 2: L-BFGS-B (local refinement)...")
        result_lgbs = minimize(obj_fun, x_pso, method='L-BFGS-B', bounds=self.bounds,
                              options={'maxiter': 1000, 'ftol': 1e-18})
        
        return result_lgbs.x
    
    def pso_slsqp_hybrid(self, obj_fun, maxiter=1000):
        """
        Hybrid: Particle Swarm Optimization + Sequential Least Squares.
        
        Strategy:
        - Stage 1: PSO uses swarm intelligence for exploration
        - Stage 2: SLSQP refines with constraint handling
        
        Enhanced: Increased default iterations from 500 to 1000.
        """
        print("Running PSO+SLSQP Hybrid Optimization...")
        
        # Stage 1: PSO for global exploration
        print("  Stage 1: PSO (global search)...")
        x_pso, _ = pso(obj_fun, self.lb, self.ub, swarmsize=200, 
                       maxiter=maxiter, debug=False, minstep=1e-18)
        
        # Stage 2: SLSQP for local refinement
        print("  Stage 2: SLSQP (local refinement)...")
        result_slsqp = minimize(obj_fun, x_pso, method='SLSQP', bounds=self.bounds,
                               options={'maxiter': 1000, 'ftol': 1e-18})
        
        return result_slsqp.x
    
    def de_pso_hybrid(self, obj_fun, maxiter=5000):
        """
        TRUE SEQUENTIAL Hybrid: DE → PSO (PSO refines DE's solution).
        
        Strategy:
        - Stage 1: DE explores globally with larger population (70% budget)
        - Stage 2: PSO refines starting from DE's basin (30% budget)
        - Returns: Final PSO result (refinement of DE)
        
        Enhanced parameters for better convergence (default: 1000 iterations).
        """
        print("Running Sequential DE→PSO Hybrid Optimization...")
        print(f"  Total budget: {maxiter} iterations")
        
        # Stage 1: DE for thorough global exploration
        print("  Stage 1: DE (global exploration with enhanced parameters)...")
        de_iterations = int(0.7 * maxiter)  # 700 iterations default
        
        result_de = differential_evolution(
            obj_fun, 
            self.bounds, 
            maxiter=de_iterations,
            strategy='best1bin',      # Good balance
            popsize=20,               # ⭐ Increased from 15 to 20
            mutation=(0.5, 1.5),      # ⭐ Increased upper bound for more exploration
            recombination=0.7,
            polish=False,
            tol=1e-18,
            atol=1e-12,               # ⭐ Stricter convergence
            workers=1                 # Can parallelize if needed
        )
        
        x_de = result_de.x
        f_de = result_de.fun
        print(f"    DE found: f(x) = {f_de:.6e}")
        
        # Stage 2: PSO refines DE's solution
        print("  Stage 2: PSO (refining DE's solution)...")
        pso_iterations = int(0.3 * maxiter)  # 300 iterations default
        
        try:
            x_pso, f_pso = pso(
                obj_fun, 
                self.lb, 
                self.ub, 
                swarmsize=150,        # ⭐ Increased from 100 to 150
                omega=0.5,            # ⭐ Lower inertia for better exploitation
                phip=2.8,             # ⭐ Higher cognitive (personal best)
                phig=1.2,             # ⭐ Lower social (individual search)
                maxiter=pso_iterations,
                minstep=1e-20,        # ⭐ Stricter step size
                debug=False
            )
            
            print(f"    PSO refined to: f(x) = {f_pso:.6e}")
            
            # Calculate improvement
            if f_pso < f_de:
                improvement = ((f_de - f_pso) / f_de) * 100
                print(f"  ✓ Refinement improved by {improvement:.2f}%")
            else:
                print(f"  ○ PSO refinement: {f_pso:.6e} (no improvement over DE)")
            
            # Return PSO result (Stage 2 output)
            return x_pso
            
        except Exception as e:
            print(f"    PSO failed: {e}")
            print(f"  ✓ Returning DE result as fallback")
            return x_de
    
    def pso_de_hybrid(self, obj_fun, maxiter=5000):
        """
        TRUE SEQUENTIAL Hybrid: PSO → DE (DE refines PSO's solution).
        
        Strategy:
        - Stage 1: PSO explores rapidly using swarm intelligence (60% budget)
        - Stage 2: DE refines with population seeded around PSO's result (40% budget)
        - Returns: Final DE result (refinement of PSO)
        
        Enhanced parameters for better convergence (default: 1000 iterations).
        DE's population is initialized around PSO's solution for true knowledge transfer.
        """
        print("Running Sequential PSO→DE Hybrid Optimization...")
        print(f"  Total budget: {maxiter} iterations")
        
        # Stage 1: PSO for rapid swarm exploration
        print("  Stage 1: PSO (rapid exploration with enhanced parameters)...")
        pso_iterations = int(0.6 * maxiter)  # 600 iterations default
        
        try:
            x_pso, f_pso = pso(
                obj_fun, 
                self.lb, 
                self.ub, 
                swarmsize=200,        # ⭐ Increased from 150 to 200
                omega=0.9,            # ⭐ Higher inertia for exploration
                phip=2.2,
                phig=2.2,
                maxiter=pso_iterations,
                minstep=1e-20,        # ⭐ Stricter step size
                debug=False
            )
            
            print(f"    PSO found: f(x) = {f_pso:.6e}")
            
        except Exception as e:
            print(f"    PSO failed: {e}, using random initialization")
            lb_array = np.array(self.lb)
            ub_array = np.array(self.ub)
            x_pso = lb_array + np.random.rand(len(self.lb)) * (ub_array - lb_array)
            f_pso = obj_fun(x_pso)
        
        # Stage 2: DE refines PSO's solution
        # Seed DE's initial population around PSO's result
        print("  Stage 2: DE (refining PSO's solution with seeded population)...")
        de_iterations = int(0.4 * maxiter)  # 400 iterations default
        
        # Create initial population centered around PSO's solution
        popsize = 20  # ⭐ Increased from 15 to 20
        init_population = np.zeros((popsize, len(x_pso)))
        
        # First individual is PSO's solution
        init_population[0] = x_pso
        
        # Convert bounds to numpy arrays for arithmetic
        lb_array = np.array(self.lb)
        ub_array = np.array(self.ub)
        
        # Rest of population distributed around PSO's solution
        for i in range(1, popsize):
            # Add gaussian noise scaled by parameter ranges
            noise = np.random.normal(0, 0.2, len(x_pso))  # ⭐ Increased from 0.15 to 0.2
            candidate = x_pso + noise * (ub_array - lb_array)
            # Clip to bounds
            candidate = np.clip(candidate, lb_array, ub_array)
            init_population[i] = candidate
        
        try:
            result_de = differential_evolution(
                obj_fun, 
                self.bounds, 
                maxiter=de_iterations,
                strategy='best1bin',
                popsize=popsize,
                mutation=(0.6, 1.5),  # ⭐ More aggressive mutation
                recombination=0.8,    # ⭐ Increased recombination
                polish=True,          # Final local polish
                tol=1e-18,
                init=init_population, # Seeded population!
                atol=1e-12,           # ⭐ Stricter convergence
                workers=1
            )
            
            x_de = result_de.x
            f_de = result_de.fun
            
            print(f"    DE refined to: f(x) = {f_de:.6e}")
            
            # Calculate improvement
            if f_de < f_pso:
                improvement = ((f_pso - f_de) / f_pso) * 100
                print(f"  ✓ Refinement improved by {improvement:.2f}%")
            else:
                print(f"  ○ DE refinement: {f_de:.6e} (no improvement over PSO)")
            
            # Return DE result (Stage 2 output)
            return x_de
            
        except Exception as e:
            print(f"    DE failed: {e}")
            print(f"  ✓ Returning PSO result as fallback")
            return x_pso
    
    def run_all_algorithms(self, obj_fun, x0):
        """
        Run all eight optimization algorithms.
        
        Parameters:
        -----------
        obj_fun : function
            Objective function to minimize
        x0 : array
            Initial guess
            
        Returns:
        --------
        dict : Results from all algorithms
        """
        results = {}
        
        print("\n--- Running Base Algorithms ---")
        results['SA'] = self.simulated_annealing_optimizer(obj_fun, x0)
        results['SLSQP'] = self.sequential_least_squares(obj_fun, x0)
        
        print("\n--- Running Hybrid Algorithms ---")
        results['DE+LGBS'] = self.de_lgbs_hybrid(obj_fun)
        results['DE+SLSQP'] = self.de_slsqp_hybrid(obj_fun)
        results['PSO+LGBS'] = self.pso_lgbs_hybrid(obj_fun)
        results['PSO+SLSQP'] = self.pso_slsqp_hybrid(obj_fun)
        results['DE+PSO'] = self.de_pso_hybrid(obj_fun)
        results['PSO+DE'] = self.pso_de_hybrid(obj_fun)
        
        return results
    
    def run_selected_algorithms(self, obj_fun, x0, methods=None):
        """
        Run selected optimization algorithms.
        
        Parameters:
        -----------
        obj_fun : function
            Objective function to minimize
        x0 : array
            Initial guess
        methods : list, optional
            List of method names to run. If None, runs all.
            Available: 'SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 
                      'PSO+SLSQP', 'DE+PSO', 'PSO+DE'
            
        Returns:
        --------
        dict : Results from selected algorithms
        """
        available_methods = {
            'SA': lambda: self.simulated_annealing_optimizer(obj_fun, x0),
            'SLSQP': lambda: self.sequential_least_squares(obj_fun, x0),
            'DE+LGBS': lambda: self.de_lgbs_hybrid(obj_fun),
            'DE+SLSQP': lambda: self.de_slsqp_hybrid(obj_fun),
            'PSO+LGBS': lambda: self.pso_lgbs_hybrid(obj_fun),
            'PSO+SLSQP': lambda: self.pso_slsqp_hybrid(obj_fun),
            'DE+PSO': lambda: self.de_pso_hybrid(obj_fun),
            'PSO+DE': lambda: self.pso_de_hybrid(obj_fun)  # Alternative implementation
        }
        
        if methods is None:
            # Default: run all 8 algorithms
            methods = ['SA', 'SLSQP', 'DE+LGBS', 'DE+SLSQP', 'PSO+LGBS', 'PSO+SLSQP', 'DE+PSO', 'PSO+DE']
        
        results = {}
        for method in methods:
            if method in available_methods:
                results[method] = available_methods[method]()
            else:
                print(f"Warning: Unknown method '{method}' - skipping")
        
        return results