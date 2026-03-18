"""
Circuit Models Module
Contains five circuit models for impedance fitting.
"""

import numpy as np


class CircuitModels:
    """
    Class containing circuit models for impedance fitting.
    """
    
    def __init__(self, model_type):
        """
        Initialize circuit model.
        
        Parameters:
        -----------
        model_type : str
            'piecewise', 'unified', 'rs_c', 'rs_cpe', or 'gcpe_series'
        """
        self.model_type = model_type
        self.omega_0 = 2 * np.pi * 1e6  # Reference frequency (1 MHz)
        
        if model_type == 'piecewise':
            self._setup_piecewise_model()
        elif model_type == 'unified':
            self._setup_unified_model()
        elif model_type == 'rs_c':
            self._setup_rs_c_model()
        elif model_type == 'rs_cpe':
            self._setup_rs_cpe_model()
        elif model_type == 'gcpe_series':
            self._setup_gcpe_series_model()
        else:
            raise ValueError("model_type must be 'piecewise', 'unified', 'rs_c', 'rs_cpe', or 'gcpe_series'")
    
    def _setup_piecewise_model(self):
        """Setup piecewise model parameters."""
        # Parameters: [Rs, A, α, B, β, Rs_h, L, Ch]
        self.param_names = ['Rs (Ω)', 'A (S)', 'α', 'B (F)', 'β', 'Rs_high (Ω)', 'L (H)', 'C_high (F)']
        self.lb = [0.01, 1e-12, 0.1, 1e-12, 0.1, 0.1, 1e-9, 1e-13]
        self.ub = [10.0, 1e-6,  2.0, 1e-6,  2.0, 1e4, 1e-3, 1e-9]
        self.bounds = list(zip(self.lb, self.ub))
        self.kink_freq = 5.2e6
    
    def _setup_unified_model(self):
        """Setup unified model: Rs + CPE + GCPE (simple 2-param formulation)."""
        # Parameters: [Rs, A, α, B, β]
        self.param_names = ['Rs (Ω)', 'A (F)', 'α', 'B (S)', 'β']
        self.lb = [0.001, 1e-16, 0.01, 1e-16, 0.01]
        self.ub = [10.0, 1e-6,  2.0, 1e-3,  2.0]
        self.bounds = list(zip(self.lb, self.ub))
    
    def _setup_rs_c_model(self):
        """Setup Rs + C model parameters."""
        # Parameters: [Rs, C]
        self.param_names = ['Rs (Ω)', 'C (F)']
        self.lb = [0.01, 1e-13]
        self.ub = [1e4, 1e-6]
        self.bounds = list(zip(self.lb, self.ub))
    
    def _setup_rs_cpe_model(self):
        """Setup Rs + CPE model parameters."""
        # Parameters: [Rs, A, α, B, β]
        self.param_names = ['Rs (Ω)', 'A (S)', 'α', 'B (F)', 'β']
        self.lb = [0.01, 1e-12, 0.1, 1e-12, 0.1]
        self.ub = [1e4, 1e-6,  2.0, 1e-6,  2.0]
        self.bounds = list(zip(self.lb, self.ub))
    
    def _setup_gcpe_series_model(self):
        """Setup gcpe_series: Rs + CPE + GCPE (four-param formulation)."""
        # Parameters: [Rs, A, α, B, β, C, γ, D, δ]
        self.param_names = ['Rs (Ω)', 'A (S)', 'α', 'B (F)', 'β', 'C (S)', 'γ', 'D (F)', 'δ']
        self.lb = [0.01, 1e-12, 0.1, 1e-12, 0.1, 1e-12, 0.1, 1e-12, 0.1]
        self.ub = [1e4, 1e-6, 2.0, 1e-6, 2.0, 1e-6, 2.0, 1e-6, 2.0]
        self.bounds = list(zip(self.lb, self.ub))
    
    def compute_initial_params(self, freq, Zf, phase_deg):
        """
        Compute initial parameter estimates.
        
        Parameters:
        -----------
        freq : array
            Frequency data
        Zf : array
            Complex impedance data
        phase_deg : array
            Phase in degrees
            
        Returns:
        --------
        list : Initial parameter estimates
        """
        if self.model_type == 'piecewise':
            return self._compute_piecewise_initial_params(freq, Zf, phase_deg)
        elif self.model_type == 'unified':
            return self._compute_unified_initial_params(freq, Zf, phase_deg)
        elif self.model_type == 'rs_c':
            return self._compute_rs_c_initial_params(freq, Zf)
        elif self.model_type == 'rs_cpe':
            return self._compute_rs_cpe_initial_params(freq, Zf)
        elif self.model_type == 'gcpe_series':
            return self._compute_gcpe_series_initial_params(freq, Zf)
    
    def _compute_piecewise_initial_params(self, freq, Zf, phase_deg):
        """Compute initial parameters for piecewise model."""
        Rs_est = np.real(np.mean(Zf[freq > 5e6]))
        A_est = 1e-9
        alpha_est = 0.8
        B_est = 1e-9
        beta_est = 0.9
        Rs_h = Rs_est

        phase_unwrap = np.unwrap(np.deg2rad(phase_deg[freq > 5e6]))
        freq_high_sub = freq[freq > 5e6]
        if len(freq_high_sub) > 1:
            slope_phase = np.diff(phase_unwrap) / np.diff(freq_high_sub)
            L_est = np.abs(np.mean(slope_phase)) / (2 * np.pi)
        else:
            L_est = 1e-6

        Ch_est = 1e-12
        p0 = [Rs_est, A_est, alpha_est, B_est, beta_est, Rs_h, L_est, Ch_est]
        
        print(f"Initial Guesses (Piecewise CPE Model):")
        print(f"Rs={Rs_est:.2f} Ω, A={A_est:.2e} S, α={alpha_est:.2f}")
        print(f"B={B_est:.2e} F, β={beta_est:.2f}")
        print(f"Rs_h={Rs_h:.2f} Ω, L={L_est:.2e} H, Ch={Ch_est:.2e} F")
        
        return p0
    
    def _compute_unified_initial_params(self, freq, Zf, phase_deg):
        """Compute initial parameters for unified model (5 params - simple formulation)."""
        Rs_est = np.real(np.mean(Zf[freq > 1e6]))
        A_est = 1e-9
        alpha_est = 0.8
        B_est = 1e-9
        beta_est = 0.9

        p0 = [Rs_est, A_est, alpha_est, B_est, beta_est]
        
        print(f"Initial Guesses (Unified - Simple Formulation):")
        print(f"Rs={Rs_est:.2f} Ω")
        print(f"CPE: A={A_est:.2e} F, α={alpha_est:.2f}")
        print(f"GCPE: B={B_est:.2e} S, β={beta_est:.2f}")
        
        return p0
    
    def _compute_rs_c_initial_params(self, freq, Zf):
        """Compute initial parameters for Rs + C model."""
        Rs_est = np.real(np.mean(Zf[freq > 1e6]))
        # Estimate C from low frequency impedance
        Z_low = np.abs(Zf[freq < 1e5])
        f_low = freq[freq < 1e5]
        if len(Z_low) > 0 and len(f_low) > 0:
            C_est = 1 / (2 * np.pi * f_low[0] * Z_low[0])
        else:
            C_est = 1e-9
        
        C_est = np.clip(C_est, 1e-13, 1e-6)
        p0 = [Rs_est, C_est]
        
        print(f"Initial Guesses (Rs + C Model):")
        print(f"Rs={Rs_est:.2f} Ω, C={C_est:.2e} F")
        
        return p0
    
    def _compute_rs_cpe_initial_params(self, freq, Zf):
        """Compute initial parameters for Rs + CPE model."""
        Rs_est = np.real(np.mean(Zf[freq > 1e6]))
        A_est = 1e-9
        alpha_est = 0.8
        B_est = 1e-9
        beta_est = 0.9
        
        p0 = [Rs_est, A_est, alpha_est, B_est, beta_est]
        
        print(f"Initial Guesses (Rs + CPE Model):")
        print(f"Rs={Rs_est:.2f} Ω, A={A_est:.2e} S, α={alpha_est:.2f}")
        print(f"B={B_est:.2e} F, β={beta_est:.2f}")
        
        return p0
    
    def _compute_gcpe_series_initial_params(self, freq, Zf):
        """Compute initial parameters for GCPE series model (9 params - four-param formulation)."""
        Rs_est = np.real(np.mean(Zf[freq > 1e6]))
        A_est = 1e-9
        alpha_est = 0.8
        B_est = 1e-9
        beta_est = 0.9
        C_est = 1e-10
        gamma_est = 0.5
        D_est = 1e-11
        delta_est = 0.7
        
        p0 = [Rs_est, A_est, alpha_est, B_est, beta_est, C_est, gamma_est, D_est, delta_est]
        
        print(f"Initial Guesses (GCPE Series - Four-Param Formulation):")
        print(f"Rs={Rs_est:.2f} Ω")
        print(f"CPE: A={A_est:.2e} S, α={alpha_est:.2f}, B={B_est:.2e} F, β={beta_est:.2f}")
        print(f"GCPE: C={C_est:.2e} S, γ={gamma_est:.2f}, D={D_est:.2e} F, δ={delta_est:.2f}")
        
        return p0
    
    def split_frequency_regions(self, freq, Zf):
        """
        Split data into frequency regions (for piecewise model).
        
        Parameters:
        -----------
        freq : array
            Frequency data
        Zf : array
            Complex impedance data
            
        Returns:
        --------
        tuple : Frequency region data
        """
        if self.model_type == 'piecewise':
            idx_low = freq <= self.kink_freq
            idx_high = freq > self.kink_freq
            
            freq_low = freq[idx_low]
            freq_high = freq[idx_high]
            Z_low = Zf[idx_low]
            Z_high = Zf[idx_high]
            
            return freq_low, freq_high, Z_low, Z_high
        else:
            # For all other models, return full data
            return freq, freq, Zf, Zf
    
    def model_low_frequency(self, p, freq):
        """Low frequency model with two-parameter CPE."""
        w = 2 * np.pi * freq
        Rs, A, alpha, B, beta = p[0], p[1], p[2], p[3], p[4]
        
        G_cpe = A * (w / self.omega_0) ** alpha
        C_cpe = B * (w / self.omega_0) ** beta
        Y_cpe = G_cpe + 1j * w * C_cpe
        Z_cpe = 1 / Y_cpe
        
        return Rs + Z_cpe
    
    def model_high_frequency(self, p, freq):
        """High frequency model with RLC."""
        w = 2 * np.pi * freq
        Rs_h, L, Ch = p[5], p[6], p[7]
        return Rs_h + 1j * w * L + 1 / (1j * w * Ch)
    
    def model_unified(self, p, freq):
        """Unified model: Rs + CPE + GCPE (simple 2-param formulation)."""
        w = 2 * np.pi * freq
        Rs, A, alpha, B, beta = p
        
        # CPE: Z = 1/(jω·A(ω/ω₀)^α)
        Z_cpe = 1 / (1j * w * A * (w / self.omega_0) ** alpha)
        
        # GCPE: Z = 1/(B(ω/ω₀)^β)
        Z_gcpe = 1 / (B * (w / self.omega_0) ** beta)
        
        return Rs + Z_cpe + Z_gcpe
    
    def model_rs_c(self, p, freq):
        """Rs + C model."""
        w = 2 * np.pi * freq
        Rs, C = p
        return Rs + 1 / (1j * w * C)
    
    def model_rs_cpe(self, p, freq):
        """Rs + CPE model."""
        w = 2 * np.pi * freq
        Rs, A, alpha, B, beta = p
        
        # CPE
        G_cpe = A * (w / self.omega_0) ** alpha
        C_cpe = B * (w / self.omega_0) ** beta
        Y_cpe = G_cpe + 1j * w * C_cpe
        Z_cpe = 1 / Y_cpe
        
        return Rs + Z_cpe
    
    def model_gcpe_series(self, p, freq):
        """GCPE Series: Rs + CPE + GCPE (four-param formulation)."""
        w = 2 * np.pi * freq
        Rs, A, alpha, B, beta, C, gamma, D, delta = p
        
        # CPE: Y_CPE = A(ω/ω₀)^α + jωB(ω/ω₀)^β
        G_cpe = A * (w / self.omega_0) ** alpha  # CPE conductance component
        C_cpe = B * (w / self.omega_0) ** beta   # CPE capacitance component
        Y_cpe = G_cpe + 1j * w * C_cpe           # Total CPE admittance
        Z_cpe = 1 / Y_cpe                        # CPE impedance
        
        # GCPE: Y_GCPE = C(ω/ω₀)^γ + jωD(ω/ω₀)^δ  
        G_gcpe = C * (w / self.omega_0) ** gamma  # GCPE conductance component
        C_gcpe = D * (w / self.omega_0) ** delta  # GCPE capacitance component
        Y_gcpe = G_gcpe + 1j * w * C_gcpe         # Total GCPE admittance
        Z_gcpe = 1 / Y_gcpe                       # GCPE impedance
        
        return Rs + Z_cpe + Z_gcpe
    
    def compute_model_impedance(self, p, freq_data):
        """
        Compute model impedance based on model type.
        
        Parameters:
        -----------
        p : array
            Model parameters
        freq_data : tuple
            Frequency data
            
        Returns:
        --------
        array : Model impedance
        """
        if self.model_type == 'piecewise':
            freq_low, freq_high, _, _ = freq_data
            Z_low = self.model_low_frequency(p, freq_low)
            Z_high = self.model_high_frequency(p, freq_high)
            return np.concatenate([Z_low, Z_high])
        elif self.model_type == 'unified':
            freq, _, _, _ = freq_data
            return self.model_unified(p, freq)
        elif self.model_type == 'rs_c':
            freq, _, _, _ = freq_data
            return self.model_rs_c(p, freq)
        elif self.model_type == 'rs_cpe':
            freq, _, _, _ = freq_data
            return self.model_rs_cpe(p, freq)
        elif self.model_type == 'gcpe_series':
            freq, _, _, _ = freq_data
            return self.model_gcpe_series(p, freq)
    
    def get_model_description(self):
        """Get model description string."""
        descriptions = {
            'piecewise': "Piecewise Model: CPE || GCPE + Rs (low-f) and R + L + C (high-f)",
            'unified': "Unified Model: CPE + GCPE for entire frequency range",
            'rs_c': "Rs + C Model: Series resistance and capacitance",
            'rs_cpe': "Rs + CPE Model: Series resistance and constant phase element",
            'gcpe_series': "GCPE||CPE Series Model: Two parallel blocks in series with Rs"
        }
        return descriptions.get(self.model_type, "Unknown model")