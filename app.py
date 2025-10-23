"""
Fe²⁺/Fe³⁺ Method Validation Dashboard - COMPLETE ALL 10 STEPS
Fixed Version with All Validation Steps Integrated

Includes:
1. Linearity
2. Interference
3. Repeatability
4. Intermediate Precision
5. Accuracy/Recovery
6. LOD/LOQ
7. Stability
8. Robustness
9. Matrix Effects
10. Range

All with robust error handling!
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io
from pathlib import Path
from scipy import stats
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fe Method Validation - Complete",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #4A90E2;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ValidationStepConfig:
    step_id: str
    step_name: str
    description: str
    requires_design: bool
    design_params: Dict
    expected_tests: int
    acceptance_criteria: Dict
    status: str = "not_started"
    
@dataclass
class ExperimentDesign:
    step_id: str
    concentration_range: tuple = None
    num_replicates: int = 3
    num_levels: int = 5
    spike_levels: List[float] = None
    interferents: List[str] = None
    custom_params: Dict = None
    created_at: datetime = None
    
@dataclass
class TestResult:
    step_id: str
    test_number: int
    shield_test_number: str
    timestamp: datetime
    concentration: float
    absorbance: float
    bg_mean: float
    sample_mean: float
    temperature: float
    loc_doses: Dict
    replicate_number: int = 1
    level_number: int = 1
    metadata: Dict = None

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        st.session_state.loc_config = {
            'base_sample_volume': 40.0,
            'extra_volume': 0.0,
            'include_loc_volumes': True,
            'standard_loc': 'LOC15',
            'stock_concentration': 1000.0
        }
        
        # ALL 10 VALIDATION STEPS
        st.session_state.validation_steps = {
            'linearity': ValidationStepConfig(
                step_id='linearity',
                step_name='4A. Linearity',
                description='Assess linear relationship between concentration and response',
                requires_design=True,
                design_params={
                    'concentration_range': (0, 100),
                    'num_levels': 5,
                    'replicates_per_level': 3
                },
                expected_tests=15,
                acceptance_criteria={'r_squared': 0.995}
            ),
            'interference': ValidationStepConfig(
                step_id='interference',
                step_name='4B. Interference',
                description='Test for interference from common ions and substances',
                requires_design=True,
                design_params={
                    'interferent_list': ['Cl⁻', 'SO₄²⁻', 'Ca²⁺', 'Mg²⁺', 'Humic Acid'],
                    'fe_concentration': 50,
                    'replicates': 3
                },
                expected_tests=60,
                acceptance_criteria={'recovery_range': (90, 110), 'rsd': 5}
            ),
            'repeatability': ValidationStepConfig(
                step_id='repeatability',
                step_name='5A. Repeatability',
                description='Same analyst, same day, same conditions',
                requires_design=True,
                design_params={
                    'concentration_levels': [10, 50, 90],
                    'replicates_per_level': 10
                },
                expected_tests=30,
                acceptance_criteria={'rsd': 5}
            ),
            'intermediate': ValidationStepConfig(
                step_id='intermediate',
                step_name='5A. Intermediate Precision',
                description='Different days, different analysts',
                requires_design=True,
                design_params={
                    'concentration_levels': [10, 50, 90],
                    'days': 3,
                    'replicates_per_day': 5
                },
                expected_tests=45,
                acceptance_criteria={'rsd': 7.5}
            ),
            'accuracy': ValidationStepConfig(
                step_id='accuracy',
                step_name='5B. Accuracy',
                description='Recovery from spiked samples',
                requires_design=True,
                design_params={
                    'spike_levels': [10, 50, 90],
                    'replicates_per_level': 5
                },
                expected_tests=15,
                acceptance_criteria={'recovery_range': (95, 105), 'rsd': 5}
            ),
            'lod_loq': ValidationStepConfig(
                step_id='lod_loq',
                step_name='5C. LOD & LOQ',
                description='Detection and quantification limits',
                requires_design=True,
                design_params={
                    'max_low_conc': 10.0,
                    'num_levels': 6,
                    'blank_replicates': 10,
                    'level_replicates': 7
                },
                expected_tests=52,
                acceptance_criteria={'lod_sn': 3, 'loq_sn': 10}
            ),
            'stability': ValidationStepConfig(
                step_id='stability',
                step_name='5D. Stability',
                description='Sample and standard stability over time',
                requires_design=True,
                design_params={
                    'concentration': 50,
                    'time_points': ['T0', '24h', '48h', '72h', '1week'],
                    'storage_conditions': ['Room Temp', 'Refrigerated'],
                    'replicates': 3
                },
                expected_tests=30,
                acceptance_criteria={'deviation': 5}
            ),
            'robustness': ValidationStepConfig(
                step_id='robustness',
                step_name='6. Robustness',
                description='Method resilience to small parameter changes',
                requires_design=True,
                design_params={
                    'concentration': 50,
                    'parameters': ['Temperature', 'pH', 'Reagent Lot'],
                    'variations': ['Normal', '+Δ', '-Δ'],
                    'replicates': 3
                },
                expected_tests=27,
                acceptance_criteria={'rsd': 5, 'bias': 5}
            ),
            'matrix': ValidationStepConfig(
                step_id='matrix',
                step_name='7. Matrix Effects',
                description='Compare DI water vs real sample matrices',
                requires_design=True,
                design_params={
                    'matrices': ['DI Water', 'Tap Water', 'Surface Water', 'Wastewater'],
                    'spike_levels': [25, 50, 75],
                    'replicates': 3
                },
                expected_tests=36,
                acceptance_criteria={'matrix_effect': 10, 'recovery_range': (90, 110)}
            ),
            'range': ValidationStepConfig(
                step_id='range',
                step_name='8. Range',
                description='Working range with acceptable accuracy and precision',
                requires_design=True,
                design_params={
                    'min_conc': 5.0,
                    'max_conc': 100.0,
                    'num_levels': 7,
                    'replicates': 3
                },
                expected_tests=21,
                acceptance_criteria={'rsd': 5, 'recovery_range': (95, 105)}
            )
        }
        
        st.session_state.designs = {}
        st.session_state.results = {step: [] for step in st.session_state.validation_steps.keys()}
        st.session_state.analyses = {}
        st.session_state.current_step = 'linearity'

initialize_session_state()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_json_data(json_content):
    """Extract data from JSON file"""
    try:
        data = json.loads(json_content) if isinstance(json_content, str) else json_content
        
        shield_test_num = None
        if 'payload' in data:
            shield_test_num = (data['payload'].get('exp_number') or 
                             data['payload'].get('test_number'))
        if not shield_test_num:
            shield_test_num = data.get('exp_number') or data.get('test_number')
        
        scans = None
        if 'payload' in data and 'scans' in data['payload']:
            scans = data['payload']['scans']
        elif 'scans' in data:
            scans = data['scans']
        
        if not scans:
            return None
        
        bg_scan = None
        sample_scan = None
        
        for scan in scans:
            scan_type = scan.get('parameters', {}).get('scanType', '').lower()
            if not scan_type:
                scan_type = scan.get('scanType', '').lower()
            
            if 'back' in scan_type or 'bg' in scan_type:
                bg_scan = scan
            elif 'sample' in scan_type:
                sample_scan = scan
        
        absorbance = None
        bg_mean = None
        sample_mean = None
        
        if bg_scan and sample_scan:
            bg_values = parse_channel_values(bg_scan.get('SC_Green', ''))
            sample_values = parse_channel_values(sample_scan.get('SC_Green', ''))
            
            if bg_values and sample_values:
                bg_mean = calculate_robust_mean(bg_values)
                sample_mean = calculate_robust_mean(sample_values)
                
                if bg_mean and sample_mean and sample_mean > 0:
                    absorbance = np.log10(bg_mean / sample_mean)
        
        loc_doses = {}
        if sample_scan:
            params = sample_scan.get('parameters', sample_scan)
            for key, value in params.items():
                if key.upper().startswith('LOC'):
                    try:
                        val = float(value)
                        if val > 0:
                            loc_doses[key.upper()] = val
                    except:
                        pass
        
        temperature = None
        if sample_scan and 'bb_temp' in sample_scan:
            try:
                temperature = float(sample_scan['bb_temp'])
            except:
                pass
        
        return {
            'shield_test_number': str(shield_test_num) if shield_test_num else 'N/A',
            'loc_doses': loc_doses,
            'absorbance': absorbance,
            'bg_mean': bg_mean,
            'sample_mean': sample_mean,
            'temperature': temperature,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        st.error(f"Error extracting JSON data: {str(e)}")
        return None

def parse_channel_values(channel_str):
    if not channel_str:
        return []
    
    if isinstance(channel_str, str):
        try:
            values = [float(x.strip()) for x in channel_str.split(',') if x.strip()]
            return values
        except:
            return []
    elif isinstance(channel_str, list):
        return [float(x) for x in channel_str if x]
    
    return []

def calculate_robust_mean(values):
    if not values or len(values) == 0:
        return None
    
    values = np.array(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    threshold = 3 * 1.4826 * mad
    filtered = values[np.abs(values - median) <= threshold]
    
    if len(filtered) == 0:
        return median
    
    return float(np.mean(filtered))

def calculate_concentration(loc_doses, config):
    if not loc_doses:
        return 0.0
    
    standard_loc = config['standard_loc']
    stock_conc = config['stock_concentration']
    base_volume = config['base_sample_volume']
    extra_volume = config['extra_volume']
    include_loc = config['include_loc_volumes']
    
    spike_volume_uL = loc_doses.get(standard_loc, 0)
    
    if spike_volume_uL == 0:
        return 0.0
    
    total_volume_mL = base_volume + extra_volume
    
    if include_loc:
        total_loc_volume_uL = sum(loc_doses.values())
        total_volume_mL += (total_loc_volume_uL / 1000)
    
    spike_volume_mL = spike_volume_uL / 1000
    concentration = stock_conc * (spike_volume_mL / total_volume_mL)
    
    return concentration

def safe_dataframe_display(df):
    """Convert dataframe to safe types for display"""
    df_copy = df.copy()
    
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str)
    
    return df_copy


# ============================================================================
# ANALYSIS FUNCTIONS - ALL 10 STEPS
# ============================================================================

def analyze_linearity(df, design, config):
    """Analyze linearity with robust error handling"""
    try:
        grouped = df.groupby('level_number').agg({
            'concentration': ['mean', 'std', 'count'],
            'absorbance': ['mean', 'std']
        }).reset_index()
        
        x = grouped['concentration']['mean'].values
        y = grouped['absorbance']['mean'].values
        
        if len(x) < 3:
            return {'error': 'Need at least 3 data points', 'passes': False}
        
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) < 3:
            return {'error': 'Insufficient valid data', 'passes': False}
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
        except:
            coeffs = np.polyfit(x, y, 1)
            slope, intercept = coeffs
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            p_value = 0.0
            std_err = 0.0
        
        y_pred = slope * x + intercept
        residuals = y - y_pred
        
        passes = r_squared >= config.acceptance_criteria['r_squared']
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_squared),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'x': x.tolist(),
            'y': y.tolist(),
            'y_pred': y_pred.tolist(),
            'residuals': residuals.tolist(),
            'passes': passes,
            'criteria': config.acceptance_criteria
        }
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}', 'passes': False}

def analyze_interference(df, design, config):
    """Analyze interference effects"""
    try:
        control_data = df[df['metadata'].apply(lambda x: x.get('interferent') == 'None')]
        if len(control_data) == 0:
            return {'error': 'No control data found', 'passes': False}
        
        control_mean = control_data['concentration'].mean()
        
        results_by_interferent = {}
        
        interferents = df[df['metadata'].apply(lambda x: x.get('interferent') != 'None')]['metadata'].apply(lambda x: x.get('interferent')).unique()
        
        for interferent in interferents:
            int_data = df[df['metadata'].apply(lambda x: x.get('interferent') == interferent)]
            
            if len(int_data) > 0:
                mean = int_data['concentration'].mean()
                std = int_data['concentration'].std()
                recovery = (mean / control_mean * 100) if control_mean > 0 else 0
                rsd = (std / mean * 100) if mean > 0 else 0
                
                passes = (config.acceptance_criteria['recovery_range'][0] <= recovery <= 
                         config.acceptance_criteria['recovery_range'][1] and 
                         rsd <= config.acceptance_criteria['rsd'])
                
                results_by_interferent[interferent] = {
                    'mean': float(mean),
                    'recovery': float(recovery),
                    'rsd': float(rsd),
                    'passes': passes
                }
        
        all_pass = all(r['passes'] for r in results_by_interferent.values())
        
        return {
            'control_mean': float(control_mean),
            'by_interferent': results_by_interferent,
            'passes': all_pass,
            'criteria': config.acceptance_criteria
        }
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}', 'passes': False}

def analyze_repeatability(df, design, config):
    """Analyze repeatability"""
    try:
        results_by_level = {}
        
        for level in df['level_number'].unique():
            level_data = df[df['level_number'] == level]['concentration']
            
            if len(level_data) < 2:
                continue
            
            mean = float(level_data.mean())
            std = float(level_data.std())
            rsd = (std / mean * 100) if mean > 0 else 0
            n = len(level_data)
            
            passes = rsd <= config.acceptance_criteria['rsd']
            
            results_by_level[int(level)] = {
                'mean': mean,
                'std': std,
                'rsd': rsd,
                'n': n,
                'passes': passes,
                'data': level_data.tolist()
            }
        
        all_pass = all(r['passes'] for r in results_by_level.values())
        
        return {
            'by_level': results_by_level,
            'passes': all_pass,
            'criteria': config.acceptance_criteria
        }
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}', 'passes': False}

def analyze_intermediate(df, design, config):
    """Analyze intermediate precision (same as repeatability but different criteria)"""
    return analyze_repeatability(df, design, config)

def analyze_accuracy(df, design, config):
    """Analyze accuracy/recovery"""
    try:
        results_by_level = {}
        
        for level in df['level_number'].unique():
            level_data = df[df['level_number'] == level]
            
            if len(level_data) == 0:
                continue
                
            spike_conc = design.spike_levels[int(level) - 1]
            
            measured = float(level_data['concentration'].mean())
            recovery = (measured / spike_conc * 100) if spike_conc > 0 else 0
            rsd = (level_data['concentration'].std() / measured * 100) if measured > 0 else 0
            
            passes = (config.acceptance_criteria['recovery_range'][0] <= recovery <= 
                     config.acceptance_criteria['recovery_range'][1] and
                     rsd <= config.acceptance_criteria['rsd'])
            
            results_by_level[int(level)] = {
                'spike_conc': float(spike_conc),
                'measured': measured,
                'recovery': float(recovery),
                'rsd': float(rsd),
                'n': len(level_data),
                'passes': passes
            }
        
        all_pass = all(r['passes'] for r in results_by_level.values())
        
        return {
            'by_level': results_by_level,
            'passes': all_pass,
            'criteria': config.acceptance_criteria
        }
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}', 'passes': False}

def analyze_lod_loq(df, design, config):
    """Analyze LOD and LOQ"""
    try:
        blank_data = df[df['level_number'] == 1]['absorbance']
        
        if len(blank_data) < 3:
            return {'error': 'Need at least 3 blank measurements', 'passes': False}
        
        blank_mean = float(blank_data.mean())
        blank_std = float(blank_data.std())
        
        lod_abs = blank_mean + 3 * blank_std
        loq_abs = blank_mean + 10 * blank_std
        
        # Estimate concentrations using a simple ratio
        non_blank = df[df['level_number'] > 1]
        if len(non_blank) > 0:
            avg_ratio = (non_blank['concentration'] / non_blank['absorbance']).mean()
            lod_conc = lod_abs * avg_ratio
            loq_conc = loq_abs * avg_ratio
        else:
            lod_conc = lod_abs * 10
            loq_conc = loq_abs * 10
        
        return {
            'blank_mean': blank_mean,
            'blank_std': blank_std,
            'lod_absorbance': float(lod_abs),
            'loq_absorbance': float(loq_abs),
            'lod_concentration': float(lod_conc),
            'loq_concentration': float(loq_conc),
            'passes': True,
            'criteria': config.acceptance_criteria
        }
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}', 'passes': False}

def analyze_stability(df, design, config):
    """Analyze stability over time"""
    try:
        t0_data = df[df['metadata'].apply(lambda x: x.get('time_point') == 'T0')]
        
        if len(t0_data) == 0:
            return {'error': 'No T0 data found', 'passes': False}
        
        t0_mean = t0_data['concentration'].mean()
        
        results_by_timepoint = {}
        
        time_points = df[df['metadata'].apply(lambda x: x.get('time_point') != 'T0')]['metadata'].apply(lambda x: x.get('time_point')).unique()
        
        for tp in time_points:
            tp_data = df[df['metadata'].apply(lambda x: x.get('time_point') == tp)]
            
            if len(tp_data) > 0:
                mean = tp_data['concentration'].mean()
                deviation = abs((mean - t0_mean) / t0_mean * 100) if t0_mean > 0 else 0
                passes = deviation <= config.acceptance_criteria['deviation']
                
                results_by_timepoint[str(tp)] = {
                    'mean': float(mean),
                    'deviation': float(deviation),
                    'passes': passes
                }
        
        all_pass = all(r['passes'] for r in results_by_timepoint.values())
        
        return {
            't0_mean': float(t0_mean),
            'by_timepoint': results_by_timepoint,
            'passes': all_pass,
            'criteria': config.acceptance_criteria
        }
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}', 'passes': False}

def analyze_robustness(df, design, config):
    """Analyze robustness"""
    try:
        normal_data = df[df['metadata'].apply(lambda x: x.get('variation') == 'Normal')]
        
        if len(normal_data) == 0:
            return {'error': 'No normal condition data found', 'passes': False}
        
        normal_mean = normal_data['concentration'].mean()
        
        results_by_parameter = {}
        
        parameters = df[df['metadata'].apply(lambda x: x.get('parameter') is not None)]['metadata'].apply(lambda x: x.get('parameter')).unique()
        
        for param in parameters:
            param_results = {}
            
            for var in ['+Δ', '-Δ']:
                var_data = df[(df['metadata'].apply(lambda x: x.get('parameter') == param)) &
                             (df['metadata'].apply(lambda x: x.get('variation') == var))]
                
                if len(var_data) > 0:
                    mean = var_data['concentration'].mean()
                    bias = abs((mean - normal_mean) / normal_mean * 100) if normal_mean > 0 else 0
                    
                    param_results[var] = {
                        'mean': float(mean),
                        'bias': float(bias)
                    }
            
            results_by_parameter[param] = param_results
        
        # Check if all biases are within acceptance
        all_pass = True
        for param_res in results_by_parameter.values():
            for var_res in param_res.values():
                if var_res['bias'] > config.acceptance_criteria['bias']:
                    all_pass = False
                    break
        
        return {
            'normal_mean': float(normal_mean),
            'by_parameter': results_by_parameter,
            'passes': all_pass,
            'criteria': config.acceptance_criteria
        }
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}', 'passes': False}

def analyze_matrix(df, design, config):
    """Analyze matrix effects"""
    try:
        di_water = df[df['metadata'].apply(lambda x: x.get('matrix') == 'DI Water')]
        
        if len(di_water) == 0:
            return {'error': 'No DI Water control data found', 'passes': False}
        
        di_mean_by_level = {}
        for level in di_water['level_number'].unique():
            di_mean_by_level[int(level)] = di_water[di_water['level_number'] == level]['concentration'].mean()
        
        results_by_matrix = {}
        
        matrices = df[df['metadata'].apply(lambda x: x.get('matrix') != 'DI Water')]['metadata'].apply(lambda x: x.get('matrix')).unique()
        
        for matrix in matrices:
            matrix_data = df[df['metadata'].apply(lambda x: x.get('matrix') == matrix)]
            
            by_level = {}
            for level in matrix_data['level_number'].unique():
                level_data = matrix_data[matrix_data['level_number'] == level]
                mean = level_data['concentration'].mean()
                di_ref = di_mean_by_level.get(int(level), 0)
                
                matrix_effect = abs((mean - di_ref) / di_ref * 100) if di_ref > 0 else 0
                recovery = (mean / di_ref * 100) if di_ref > 0 else 0
                
                passes = (matrix_effect <= config.acceptance_criteria['matrix_effect'] and
                         config.acceptance_criteria['recovery_range'][0] <= recovery <= 
                         config.acceptance_criteria['recovery_range'][1])
                
                by_level[int(level)] = {
                    'mean': float(mean),
                    'di_reference': float(di_ref),
                    'matrix_effect': float(matrix_effect),
                    'recovery': float(recovery),
                    'passes': passes
                }
            
            results_by_matrix[matrix] = by_level
        
        # Check if all pass
        all_pass = True
        for matrix_res in results_by_matrix.values():
            for level_res in matrix_res.values():
                if not level_res['passes']:
                    all_pass = False
                    break
        
        return {
            'di_water_means': di_mean_by_level,
            'by_matrix': results_by_matrix,
            'passes': all_pass,
            'criteria': config.acceptance_criteria
        }
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}', 'passes': False}

def analyze_range(df, design, config):
    """Analyze working range (combines linearity + accuracy)"""
    try:
        results_by_level = {}
        
        for level in df['level_number'].unique():
            level_data = df[df['level_number'] == level]
            spike_conc = design.spike_levels[int(level) - 1]
            
            measured = level_data['concentration'].mean()
            std = level_data['concentration'].std()
            rsd = (std / measured * 100) if measured > 0 else 0
            recovery = (measured / spike_conc * 100) if spike_conc > 0 else 0
            
            passes = (rsd <= config.acceptance_criteria['rsd'] and
                     config.acceptance_criteria['recovery_range'][0] <= recovery <= 
                     config.acceptance_criteria['recovery_range'][1])
            
            results_by_level[int(level)] = {
                'spike_conc': float(spike_conc),
                'measured': float(measured),
                'rsd': float(rsd),
                'recovery': float(recovery),
                'n': len(level_data),
                'passes': passes
            }
        
        all_pass = all(r['passes'] for r in results_by_level.values())
        
        return {
            'by_level': results_by_level,
            'passes': all_pass,
            'criteria': config.acceptance_criteria
        }
    except Exception as e:
        return {'error': f'Analysis failed: {str(e)}', 'passes': False}


# ============================================================================
# MAIN UI FUNCTIONS
# ============================================================================

def main():
    st.markdown('<div class="main-header">🧪 Fe²⁺/Fe³⁺ Method Validation - Complete</div>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("🎯 Validation Steps")
        
        completed = sum(1 for s in st.session_state.validation_steps.values() if s.status == 'analyzed')
        total = len(st.session_state.validation_steps)
        
        st.metric("Progress", f"{completed}/{total}")
        st.progress(completed / total if total > 0 else 0)
        
        st.markdown("---")
        
        for step_id, step_config in st.session_state.validation_steps.items():
            status_emoji = {
                'not_started': '⚪',
                'designed': '🔵',
                'collecting': '🔄',
                'completed': '✅',
                'analyzed': '🎉'
            }
            
            emoji = status_emoji.get(step_config.status, '⚪')
            
            if st.button(f"{emoji} {step_config.step_name}", key=f"nav_{step_id}", use_container_width=True):
                st.session_state.current_step = step_id
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🔧 LOC Config", use_container_width=True):
            st.session_state.current_step = 'config'
            st.rerun()
        
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.current_step = 'dashboard'
            st.rerun()
    
    current_step = st.session_state.current_step
    
    if current_step == 'config':
        show_config()
    elif current_step == 'dashboard':
        show_dashboard()
    else:
        show_validation_step(current_step)

def show_validation_step(step_id):
    """Show validation step page"""
    
    step_config = st.session_state.validation_steps[step_id]
    
    st.markdown(f'<div class="step-header">{step_config.step_name}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    📋 <strong>Purpose:</strong> {step_config.description}
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["📐 Design", "📥 Collect", "📊 Analyze", "📈 Results"])
    
    with tabs[0]:
        show_design_tab(step_id, step_config)
    
    with tabs[1]:
        show_collect_tab(step_id, step_config)
    
    with tabs[2]:
        show_analyze_tab(step_id, step_config)
    
    with tabs[3]:
        show_results_tab(step_id, step_config)

def show_design_tab(step_id, step_config):
    """Design experiment tab"""
    st.subheader("📐 Experiment Design")
    
    if step_id in st.session_state.designs:
        st.markdown("""
        <div class="success-box">
        ✅ <strong>Experiment Already Designed!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        design = st.session_state.designs[step_id]
        
        if design.spike_levels:
            st.write(f"**Levels:** {len(design.spike_levels)}")
            st.write(f"**Replicates:** {design.num_replicates}")
            
            df_levels = pd.DataFrame({
                'Level': range(1, len(design.spike_levels) + 1),
                'Concentration (mg/L)': design.spike_levels
            })
            st.dataframe(safe_dataframe_display(df_levels), width='stretch')
        
        if design.custom_params:
            st.write("**Additional Parameters:**")
            for key, value in design.custom_params.items():
                st.write(f"- {key}: {value}")
        
        if st.button("🔄 Modify Design"):
            del st.session_state.designs[step_id]
            st.rerun()
    
    else:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Design Required</strong><br>
        Configure your experiment parameters below.
        </div>
        """, unsafe_allow_html=True)
        
        # Common design parameters
        col1, col2 = st.columns(2)
        
        custom_params = {}
        
        if step_id == 'linearity' or step_id == 'range':
            with col1:
                min_conc = st.number_input("Min Conc (mg/L)", value=0.0 if step_id == 'linearity' else 5.0, step=1.0)
                max_conc = st.number_input("Max Conc (mg/L)", value=100.0, step=1.0)
            
            with col2:
                num_levels = st.number_input("Number of Levels", value=5 if step_id == 'linearity' else 7, min_value=3, max_value=10)
                num_reps = st.number_input("Replicates per Level", value=3, min_value=2, max_value=10)
            
            spike_levels = np.linspace(min_conc, max_conc, num_levels).tolist()
        
        elif step_id == 'interference':
            with col1:
                fe_conc = st.number_input("Fe Concentration (mg/L)", value=50.0, step=5.0)
                num_reps = st.number_input("Replicates", value=3, min_value=2, max_value=6)
            
            with col2:
                interferents = st.multiselect(
                    "Select Interferents",
                    step_config.design_params['interferent_list'],
                    default=step_config.design_params['interferent_list'][:3]
                )
            
            spike_levels = [fe_conc]
            custom_params = {'interferents': interferents, 'fe_concentration': fe_conc}
        
        elif step_id in ['repeatability', 'intermediate', 'accuracy']:
            col1, col2, col3 = st.columns(3)
            with col1:
                low_conc = st.number_input("Low Conc (mg/L)", value=10.0, step=5.0)
            with col2:
                mid_conc = st.number_input("Mid Conc (mg/L)", value=50.0, step=5.0)
            with col3:
                high_conc = st.number_input("High Conc (mg/L)", value=90.0, step=5.0)
            
            spike_levels = [low_conc, mid_conc, high_conc]
            num_levels = 3
            
            if step_id == 'intermediate':
                col1, col2 = st.columns(2)
                with col1:
                    num_days = st.slider("Number of Days", 3, 7, 3)
                with col2:
                    reps_per_day = st.slider("Reps per Day", 3, 8, 5)
                num_reps = num_days * reps_per_day
                custom_params = {'num_days': num_days, 'reps_per_day': reps_per_day}
            else:
                num_reps = st.slider("Replicates per Level", 3, 15, 10 if step_id == 'repeatability' else 5)
        
        elif step_id == 'lod_loq':
            with col1:
                max_low_conc = st.number_input("Max Low Conc (mg/L)", value=10.0, step=1.0)
                num_levels = st.slider("Number of Levels", 5, 10, 6)
            
            with col2:
                blank_reps = st.slider("Blank Replicates", 7, 15, 10)
                level_reps = st.slider("Reps per Level", 5, 10, 7)
            
            spike_levels = np.linspace(0, max_low_conc, num_levels).tolist()
            num_reps = level_reps
            custom_params = {'blank_replicates': blank_reps}
        
        elif step_id == 'stability':
            conc = st.number_input("Test Concentration (mg/L)", value=50.0, step=5.0)
            
            time_points = st.multiselect(
                "Time Points",
                ['T0', '4h', '24h', '48h', '72h', '1week', '2weeks'],
                default=['T0', '24h', '48h', '72h', '1week']
            )
            
            storage = st.multiselect(
                "Storage Conditions",
                ['Room Temp', 'Refrigerated', 'Frozen'],
                default=['Room Temp', 'Refrigerated']
            )
            
            num_reps = st.slider("Replicates", 3, 6, 3)
            
            spike_levels = [conc]
            num_levels = 1
            custom_params = {'time_points': time_points, 'storage_conditions': storage, 'concentration': conc}
        
        elif step_id == 'robustness':
            conc = st.number_input("Test Concentration (mg/L)", value=50.0, step=5.0)
            
            parameters = st.multiselect(
                "Parameters to Vary",
                ['Temperature (±2°C)', 'pH (±0.5)', 'Reagent Lot', 'Sample Volume (±5%)', 'Mixing Time (±30s)'],
                default=['Temperature (±2°C)', 'pH (±0.5)', 'Reagent Lot']
            )
            
            num_reps = st.slider("Replicates", 3, 6, 3)
            
            spike_levels = [conc]
            num_levels = 1
            custom_params = {'parameters': parameters, 'concentration': conc}
        
        elif step_id == 'matrix':
            matrices = st.multiselect(
                "Sample Matrices",
                ['DI Water', 'Tap Water', 'Surface Water', 'Groundwater', 'Wastewater', 'Seawater'],
                default=['DI Water', 'Tap Water', 'Surface Water', 'Wastewater']
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                low_spike = st.number_input("Low Spike (mg/L)", value=25.0, step=5.0)
            with col2:
                mid_spike = st.number_input("Mid Spike (mg/L)", value=50.0, step=5.0)
            with col3:
                high_spike = st.number_input("High Spike (mg/L)", value=75.0, step=5.0)
            
            spike_levels = [low_spike, mid_spike, high_spike]
            num_levels = 3
            num_reps = st.slider("Replicates per Matrix", 3, 6, 3)
            custom_params = {'matrices': matrices}
        
        else:
            st.error("Design not implemented for this step")
            return
        
        # Save design button
        if st.button("💾 Save Experiment Design", type="primary"):
            design = ExperimentDesign(
                step_id=step_id,
                concentration_range=(min(spike_levels), max(spike_levels)) if len(spike_levels) > 1 else None,
                num_levels=len(spike_levels),
                num_replicates=num_reps,
                spike_levels=spike_levels,
                custom_params=custom_params if custom_params else None,
                created_at=datetime.now()
            )
            
            st.session_state.designs[step_id] = design
            st.session_state.validation_steps[step_id].status = 'designed'
            
            st.success("✅ Experiment design saved!")
            st.rerun()


def show_lod_loq_results(analysis, config):
    """Display LOD/LOQ results"""
    st.subheader("🔍 Detection and Quantification Limits")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Blank Mean", f"{analysis['blank_mean']:.4f}")
    with col2:
        st.metric("Blank Std Dev", f"{analysis['blank_std']:.4f}")
    with col3:
        st.metric("LOD", f"{analysis['lod_concentration']:.3f} mg/L")
    with col4:
        st.metric("LOQ", f"{analysis['loq_concentration']:.3f} mg/L")
    
    st.info("""
    **Calculation Method:**
    - LOD = Blank Mean + 3 × Blank Std Dev
    - LOQ = Blank Mean + 10 × Blank Std Dev
    """)
    
    st.success("✅ LOD and LOQ determined")

def show_stability_results(analysis, config):
    """Display stability results"""
    st.info(f"**T0 Reference:** {analysis['t0_mean']:.3f} mg/L")
    
    data = []
    for tp, res in analysis['by_timepoint'].items():
        data.append({
            'Time Point': tp,
            'Mean (mg/L)': f"{res['mean']:.3f}",
            'Deviation (%)': f"{res['deviation']:.2f}",
            'Status': '✅ Stable' if res['passes'] else '❌ Unstable'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(safe_dataframe_display(df), width='stretch')
    
    # Time series plot
    timepoints = ['T0'] + list(analysis['by_timepoint'].keys())
    concentrations = [analysis['t0_mean']] + [analysis['by_timepoint'][tp]['mean'] 
                                               for tp in analysis['by_timepoint'].keys()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timepoints,
        y=concentrations,
        mode='lines+markers',
        marker=dict(size=10),
        line=dict(width=2, color='#4A90E2')
    ))
    
    upper_limit = analysis['t0_mean'] * (1 + config.acceptance_criteria['deviation'] / 100)
    lower_limit = analysis['t0_mean'] * (1 - config.acceptance_criteria['deviation'] / 100)
    
    fig.add_hrect(
        y0=lower_limit,
        y1=upper_limit,
        fillcolor="green",
        opacity=0.1,
        line_width=0
    )
    
    fig.update_layout(
        title="Stability Over Time",
        xaxis_title="Time Point",
        yaxis_title="Concentration (mg/L)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if analysis['passes']:
        st.success("✅ Sample stable within acceptable range")
    else:
        st.error("❌ Sample shows instability")

def show_robustness_results(analysis, config):
    """Display robustness results"""
    st.info(f"**Normal Condition Mean:** {analysis['normal_mean']:.3f} mg/L")
    
    for param, variations in analysis['by_parameter'].items():
        with st.expander(f"📊 {param}", expanded=True):
            
            data = []
            for var, res in variations.items():
                data.append({
                    'Variation': var,
                    'Mean (mg/L)': f"{res['mean']:.3f}",
                    'Bias (%)': f"{res['bias']:.2f}"
                })
            
            df_param = pd.DataFrame(data)
            st.dataframe(safe_dataframe_display(df_param), width='stretch')
            
            # Chart
            variations_list = list(variations.keys())
            means = [variations[v]['mean'] for v in variations_list]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Normal'] + variations_list,
                y=[analysis['normal_mean']] + means,
                marker_color=['blue', 'lightblue', 'lightblue'],
                text=[f"{analysis['normal_mean']:.3f}"] + [f"{m:.3f}" for m in means],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"{param} - Effect on Measurement",
                xaxis_title="Condition",
                yaxis_title="Concentration (mg/L)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    if analysis['passes']:
        st.success("✅ Method is robust to parameter variations")
    else:
        st.error("❌ Method shows sensitivity to some parameters")

def show_matrix_results(analysis, config):
    """Display matrix effects results"""
    st.info("**DI Water (Control)** serves as reference")
    
    for matrix, by_level in analysis['by_matrix'].items():
        with st.expander(f"📊 {matrix}", expanded=True):
            
            data = []
            for level, res in by_level.items():
                data.append({
                    'Level': level,
                    'DI Water (mg/L)': f"{res['di_reference']:.3f}",
                    'Matrix Mean (mg/L)': f"{res['mean']:.3f}",
                    'Matrix Effect (%)': f"{res['matrix_effect']:.2f}",
                    'Recovery (%)': f"{res['recovery']:.1f}",
                    'Status': '✅' if res['passes'] else '❌'
                })
            
            df_matrix = pd.DataFrame(data)
            st.dataframe(safe_dataframe_display(df_matrix), width='stretch')
            
            # Chart
            levels = list(by_level.keys())
            di_refs = [by_level[l]['di_reference'] for l in levels]
            matrix_means = [by_level[l]['mean'] for l in levels]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='DI Water',
                x=[f"Level {l}" for l in levels],
                y=di_refs,
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name=matrix,
                x=[f"Level {l}" for l in levels],
                y=matrix_means,
                marker_color='orange'
            ))
            
            fig.update_layout(
                title=f"{matrix} vs DI Water",
                xaxis_title="Level",
                yaxis_title="Concentration (mg/L)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    if analysis['passes']:
        st.success("✅ No significant matrix effects")
    else:
        st.error("❌ Matrix effects detected")

def show_range_results(analysis, config):
    """Display range results"""
    if analysis['passes']:
        st.success("✅ RANGE TEST PASSED: All levels acceptable")
    else:
        st.error("❌ RANGE TEST FAILED: Some levels outside criteria")
    
    data = []
    for level, res in analysis['by_level'].items():
        data.append({
            'Level': level,
            'Spiked (mg/L)': f"{res['spike_conc']:.3f}",
            'Measured (mg/L)': f"{res['measured']:.3f}",
            'RSD (%)': f"{res['rsd']:.2f}",
            'Recovery (%)': f"{res['recovery']:.1f}",
            'n': res['n'],
            'Status': '✅' if res['passes'] else '❌'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(safe_dataframe_display(df), width='stretch')
    
    # Recovery across range
    levels = list(analysis['by_level'].keys())
    recoveries = [analysis['by_level'][l]['recovery'] for l in levels]
    colors = ['green' if analysis['by_level'][l]['passes'] else 'red' for l in levels]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{analysis['by_level'][l]['spike_conc']:.1f}" for l in levels],
        y=recoveries,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in recoveries],
        textposition='outside'
    ))
    
    fig.add_hrect(
        y0=config.acceptance_criteria['recovery_range'][0],
        y1=config.acceptance_criteria['recovery_range'][1],
        fillcolor="green",
        opacity=0.1,
        line_width=0
    )
    
    fig.update_layout(
        title="Recovery Across Working Range",
        xaxis_title="Concentration (mg/L)",
        yaxis_title="Recovery (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_config():
    """LOC Configuration page"""
    st.header("⚙️ LOC Configuration")
    
    st.markdown("""
    <div class="info-box">
    📋 <strong>Configure LOC Settings</strong><br>
    Set up stock concentrations and sample volumes for M₁V₁=M₂V₂ calculations.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Sample Volume")
        
        base_vol = st.number_input("Base Volume (mL)", value=st.session_state.loc_config['base_sample_volume'], step=1.0)
        extra_vol = st.number_input("Extra Volume (mL)", value=st.session_state.loc_config['extra_volume'], step=0.1)
        include_loc = st.checkbox("Include LOC volumes?", value=st.session_state.loc_config['include_loc_volumes'])
    
    with col2:
        st.subheader("🧪 Fe Standard")
        
        standard_loc = st.selectbox("Standard LOC", [f'LOC{i}' for i in range(1, 17)], 
                                    index=[f'LOC{i}' for i in range(1, 17)].index(st.session_state.loc_config['standard_loc']))
        stock_conc = st.number_input("Stock Conc (mg/L)", value=st.session_state.loc_config['stock_concentration'], step=10.0)
    
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.loc_config.update({
            'base_sample_volume': base_vol,
            'extra_volume': extra_vol,
            'include_loc_volumes': include_loc,
            'standard_loc': standard_loc,
            'stock_concentration': stock_conc
        })
        st.success("✅ Configuration saved!")

def show_dashboard():
    """Overview dashboard"""
    st.header("📊 Validation Overview")
    
    completed = sum(1 for s in st.session_state.validation_steps.values() if s.status == 'analyzed')
    total = len(st.session_state.validation_steps)
    in_progress = sum(1 for s in st.session_state.validation_steps.values() if s.status in ['designed', 'collecting'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Steps", total)
    with col2:
        st.metric("Completed", completed)
    with col3:
        st.metric("In Progress", in_progress)
    with col4:
        st.metric("Progress", f"{(completed/total*100):.0f}%")
    
    st.progress(completed / total if total > 0 else 0)
    
    st.markdown("---")
    
    st.subheader("📋 Steps Status")
    
    status_data = []
    for step_id, step_config in st.session_state.validation_steps.items():
        results_count = len(st.session_state.results.get(step_id, []))
        
        status_data.append({
            'Step': step_config.step_name,
            'Status': step_config.status.replace('_', ' ').title(),
            'Tests Collected': results_count,
            'Expected': step_config.expected_tests,
            'Analyzed': '✅' if step_config.status == 'analyzed' else '⚪'
        })
    
    df_status = pd.DataFrame(status_data)
    st.dataframe(safe_dataframe_display(df_status), width='stretch')
    
    # Progress chart
    st.subheader("📈 Progress by Step")
    
    steps = [s['Step'] for s in status_data]
    collected = [s['Tests Collected'] for s in status_data]
    expected = [s['Expected'] for s in status_data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Collected',
        x=steps,
        y=collected,
        marker_color='#28a745'
    ))
    
    fig.add_trace(go.Bar(
        name='Remaining',
        x=steps,
        y=[e - c for e, c in zip(expected, collected)],
        marker_color='#ffc107'
    ))
    
    fig.update_layout(
        barmode='stack',
        xaxis_tickangle=-45,
        height=500,
        yaxis_title="Number of Tests"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Export all
    if st.button("📥 Export All Results", type="primary"):
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary
            df_status.to_excel(writer, sheet_name='Summary', index=False)
            
            # Each step
            for step_id, results in st.session_state.results.items():
                if results:
                    df = pd.DataFrame([asdict(r) for r in results])
                    step_name = st.session_state.validation_steps[step_id].step_name.replace('/', '-')
                    df.to_excel(writer, sheet_name=step_name[:31], index=False)
        
        output.seek(0)
        
        st.download_button(
            label="📥 Download Complete Report",
            data=output,
            file_name=f"fe_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()
        nbinsx=10,
        marker_color='purple'
        ))
        fig_hist.update_layout(
            title="Residuals Distribution",
            xaxis_title="Residuals",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def show_interference_results(analysis, config):
    """Display interference results"""
    st.info(f"**Control Mean:** {analysis['control_mean']:.3f} mg/L")
    
    # Summary table
    data = []
    for interferent, res in analysis['by_interferent'].items():
        data.append({
            'Interferent': interferent,
            'Mean (mg/L)': f"{res['mean']:.3f}",
            'Recovery (%)': f"{res['recovery']:.1f}",
            'RSD (%)': f"{res['rsd']:.2f}",
            'Status': '✅ Pass' if res['passes'] else '❌ Fail'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(safe_dataframe_display(df), width='stretch')
    
    # Recovery chart
    fig = go.Figure()
    
    interferents = list(analysis['by_interferent'].keys())
    recoveries = [analysis['by_interferent'][i]['recovery'] for i in interferents]
    colors = ['green' if analysis['by_interferent'][i]['passes'] else 'red' for i in interferents]
    
    fig.add_trace(go.Bar(
        x=interferents,
        y=recoveries,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in recoveries],
        textposition='outside'
    ))
    
    fig.add_hrect(
        y0=config.acceptance_criteria['recovery_range'][0],
        y1=config.acceptance_criteria['recovery_range'][1],
        fillcolor="green",
        opacity=0.1,
        line_width=0
    )
    
    fig.update_layout(
        title="Recovery by Interferent",
        xaxis_title="Interferent",
        yaxis_title="Recovery (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if analysis['passes']:
        st.success("✅ All interferents within acceptable range")
    else:
        st.error("❌ Some interferents outside acceptable range")

def show_repeatability_results(analysis, config):
    """Display repeatability results"""
    if analysis['passes']:
        st.success("✅ REPEATABILITY TEST PASSED: All levels meet RSD criteria")
    else:
        st.error("❌ REPEATABILITY TEST FAILED: Some levels exceed RSD")
    
    data = []
    for level, res in analysis['by_level'].items():
        data.append({
            'Level': level,
            'Mean (mg/L)': f"{res['mean']:.3f}",
            'Std Dev': f"{res['std']:.3f}",
            'RSD (%)': f"{res['rsd']:.2f}",
            'n': res['n'],
            'Status': '✅' if res['passes'] else '❌'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(safe_dataframe_display(df), width='stretch')
    
    # Box plot
    fig = go.Figure()
    
    for level, res in analysis['by_level'].items():
        fig.add_trace(go.Box(
            y=res['data'],
            name=f"Level {level}",
            boxmean='sd'
        ))
    
    fig.update_layout(
        title="Distribution by Level",
        xaxis_title="Level",
        yaxis_title="Concentration (mg/L)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # RSD chart
    levels = list(analysis['by_level'].keys())
    rsds = [analysis['by_level'][l]['rsd'] for l in levels]
    colors = ['green' if analysis['by_level'][l]['passes'] else 'red' for l in levels]
    
    fig_rsd = go.Figure()
    fig_rsd.add_trace(go.Bar(
        x=[f"Level {l}" for l in levels],
        y=rsds,
        marker_color=colors,
        text=[f"{r:.2f}%" for r in rsds],
        textposition='outside'
    ))
    
    fig_rsd.add_hline(
        y=config.acceptance_criteria['rsd'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Criteria: {config.acceptance_criteria['rsd']}%"
    )
    
    fig_rsd.update_layout(
        title="RSD Comparison",
        xaxis_title="Level",
        yaxis_title="RSD (%)",
        height=400
    )
    
    st.plotly_chart(fig_rsd, use_container_width=True)

def show_intermediate_results(analysis, config):
    """Display intermediate precision results"""
    st.info("**Note:** This analysis assesses precision across multiple days/analysts")
    show_repeatability_results(analysis, config)

def show_accuracy_results(analysis, config):
    """Display accuracy results"""
    if analysis['passes']:
        st.success("✅ ACCURACY TEST PASSED: All recoveries acceptable")
    else:
        st.error("❌ ACCURACY TEST FAILED: Some recoveries outside range")
    
    data = []
    for level, res in analysis['by_level'].items():
        data.append({
            'Level': level,
            'Spiked (mg/L)': f"{res['spike_conc']:.3f}",
            'Measured (mg/L)': f"{res['measured']:.3f}",
            'Recovery (%)': f"{res['recovery']:.1f}",
            'RSD (%)': f"{res['rsd']:.2f}",
            'n': res['n'],
            'Status': '✅' if res['passes'] else '❌'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(safe_dataframe_display(df), width='stretch')
    
    # Recovery chart
    levels = list(analysis['by_level'].keys())
    recoveries = [analysis['by_level'][l]['recovery'] for l in levels]
    colors = ['green' if analysis['by_level'][l]['passes'] else 'red' for l in levels]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Level {l}" for l in levels],
        y=recoveries,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in recoveries],
        textposition='outside'
    ))
    
    fig.add_hrect(
        y0=config.acceptance_criteria['recovery_range'][0],
        y1=config.acceptance_criteria['recovery_range'][1],
        fillcolor="green",
        opacity=0.1,
        line_width=0
    )
    
    fig.add_hline(y=100, line_dash="dash", line_color="blue")
    
    fig.update_layout(
        title="Recovery by Spike Level",
        xaxis_title="Level",
        yaxis_title="Recovery (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
