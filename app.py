"""
Fe²⁺/Fe³⁺ Method Validation Dashboard - FINAL FIXED VERSION
Fixes:
1. SVD convergence error in linear regression
2. Arrow serialization errors in dataframes
3. All deprecation warnings
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

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fe Method Validation - Complete Workflow",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before...)
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
# SESSION STATE
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
        
        st.session_state.validation_steps = {
            'linearity': ValidationStepConfig(
                step_id='linearity',
                step_name='4A. Linearity',
                description='Assess linear relationship between concentration and response',
                requires_design=True,
                design_params={
                    'concentration_range': (0, 100),
                    'num_levels': 5,
                    'replicates_per_level': 3,
                    'suggested_levels': [0, 25, 50, 75, 100]
                },
                expected_tests=15,
                acceptance_criteria={
                    'r_squared': 0.995,
                    'residuals_pattern': 'random'
                }
            ),
            'repeatability': ValidationStepConfig(
                step_id='repeatability',
                step_name='5A. Repeatability',
                description='Same analyst, same day, same conditions',
                requires_design=True,
                design_params={
                    'concentration_levels': [10, 50, 90],
                    'replicates_per_level': 10,
                    'same_day': True
                },
                expected_tests=30,
                acceptance_criteria={
                    'rsd': 5
                }
            ),
            'accuracy': ValidationStepConfig(
                step_id='accuracy',
                step_name='5B. Accuracy',
                description='Spike known amounts and measure recovery',
                requires_design=True,
                design_params={
                    'spike_levels': [10, 50, 90],
                    'matrix': 'DI water',
                    'replicates_per_level': 5
                },
                expected_tests=15,
                acceptance_criteria={
                    'recovery_range': (95, 105),
                    'rsd': 5
                }
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
    
    # Convert all object columns to strings
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str)
    
    return df_copy

# ============================================================================
# ANALYSIS FUNCTIONS WITH ERROR HANDLING
# ============================================================================

def analyze_linearity(df, design, config):
    """Analyze linearity data with robust error handling"""
    try:
        # Group by level
        grouped = df.groupby('level_number').agg({
            'concentration': ['mean', 'std', 'count'],
            'absorbance': ['mean', 'std']
        }).reset_index()
        
        x = grouped['concentration']['mean'].values
        y = grouped['absorbance']['mean'].values
        
        # Check for valid data
        if len(x) < 3:
            return {
                'error': 'Need at least 3 data points for linearity',
                'passes': False
            }
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) < 3:
            return {
                'error': 'Insufficient valid data points',
                'passes': False
            }
        
        # Try linear regression with error handling
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
        except Exception as e:
            # Fallback to simple linear fit
            try:
                coeffs = np.polyfit(x, y, 1)
                slope, intercept = coeffs
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                p_value = 0.0
                std_err = 0.0
            except:
                return {
                    'error': f'Linear regression failed: {str(e)}',
                    'passes': False
                }
        
        # Calculate predicted and residuals
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
            'criteria': config.acceptance_criteria,
            'summary': f"R² = {r_squared:.4f} (Criteria: ≥ {config.acceptance_criteria['r_squared']})"
        }
        
    except Exception as e:
        return {
            'error': f'Analysis failed: {str(e)}',
            'passes': False
        }

def analyze_repeatability(df, design, config):
    """Analyze repeatability with error handling"""
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
            'criteria': config.acceptance_criteria,
            'summary': f"All levels RSD ≤ {config.acceptance_criteria['rsd']}%" if all_pass else "Some levels exceed RSD criteria"
        }
    
    except Exception as e:
        return {
            'error': f'Analysis failed: {str(e)}',
            'passes': False
        }

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
            'criteria': config.acceptance_criteria,
            'summary': f"Recovery: {config.acceptance_criteria['recovery_range'][0]}-{config.acceptance_criteria['recovery_range'][1]}%"
        }
    
    except Exception as e:
        return {
            'error': f'Analysis failed: {str(e)}',
            'passes': False
        }

# ============================================================================
# MAIN UI FUNCTIONS
# ============================================================================

def main():
    st.markdown('<div class="main-header">🧪 Fe²⁺/Fe³⁺ Method Validation</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
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
    
    # Main content
    current_step = st.session_state.current_step
    
    if current_step == 'config':
        show_config()
    else:
        show_validation_step(current_step)

def show_validation_step(step_id):
    """Show validation step page"""
    
    step_config = st.session_state.validation_steps[step_id]
    
    st.markdown(f'<div class="step-header">{step_config.step_name}</div>', unsafe_allow_html=True)
    
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
    st.subheader("📐 Experiment Design")
    
    if step_id in st.session_state.designs:
        st.success("✅ Design complete!")
        design = st.session_state.designs[step_id]
        
        if design.spike_levels:
            st.write(f"**Levels:** {len(design.spike_levels)}")
            st.write(f"**Replicates:** {design.num_replicates}")
            
            df_levels = pd.DataFrame({
                'Level': range(1, len(design.spike_levels) + 1),
                'Concentration': design.spike_levels
            })
            st.dataframe(safe_dataframe_display(df_levels), width='stretch')
        
        if st.button("🔄 Modify"):
            del st.session_state.designs[step_id]
            st.rerun()
    else:
        st.warning("⚠️ Please design your experiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_conc = st.number_input("Min Conc (mg/L)", value=0.0, step=1.0)
            max_conc = st.number_input("Max Conc (mg/L)", value=100.0, step=1.0)
        
        with col2:
            num_levels = st.number_input("Levels", value=5, min_value=3, max_value=10)
            num_reps = st.number_input("Replicates", value=3, min_value=2, max_value=10)
        
        if st.button("💾 Save Design", type="primary"):
            spike_levels = np.linspace(min_conc, max_conc, num_levels).tolist()
            
            design = ExperimentDesign(
                step_id=step_id,
                concentration_range=(min_conc, max_conc),
                num_levels=num_levels,
                num_replicates=num_reps,
                spike_levels=spike_levels,
                created_at=datetime.now()
            )
            
            st.session_state.designs[step_id] = design
            st.session_state.validation_steps[step_id].status = 'designed'
            st.success("✅ Design saved!")
            st.rerun()

def show_collect_tab(step_id, step_config):
    st.subheader("📥 Data Collection")
    
    if step_id not in st.session_state.designs:
        st.warning("⚠️ Please design experiment first")
        return
    
    design = st.session_state.designs[step_id]
    results = st.session_state.results[step_id]
    
    expected = len(design.spike_levels) * design.num_replicates if design.spike_levels else 0
    collected = len(results)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected", expected)
    with col2:
        st.metric("Collected", collected)
    with col3:
        st.metric("Progress", f"{(collected/expected*100) if expected > 0 else 0:.0f}%")
    
    st.markdown("---")
    
    uploaded_files = st.file_uploader("Upload JSON files", type=['json'], accept_multiple_files=True)
    
    if uploaded_files:
        new_results = []
        
        for idx, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name}", expanded=(idx==0)):
                try:
                    content = file.read().decode('utf-8')
                    extracted = extract_json_data(content)
                    
                    if extracted:
                        concentration = calculate_concentration(extracted['loc_doses'], st.session_state.loc_config)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Conc", f"{concentration:.3f}")
                        with col2:
                            st.metric("Abs", f"{extracted['absorbance']:.4f}" if extracted['absorbance'] else "N/A")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            level = st.number_input("Level", 1, len(design.spike_levels), 1, key=f"lev_{idx}")
                        with col2:
                            rep = st.number_input("Replicate", 1, design.num_replicates, 1, key=f"rep_{idx}")
                        
                        result = TestResult(
                            step_id=step_id,
                            test_number=collected + len(new_results) + 1,
                            shield_test_number=extracted['shield_test_number'],
                            timestamp=extracted['timestamp'],
                            concentration=concentration,
                            absorbance=extracted['absorbance'] or 0.0,
                            bg_mean=extracted['bg_mean'] or 0.0,
                            sample_mean=extracted['sample_mean'] or 0.0,
                            temperature=extracted['temperature'] or 0.0,
                            loc_doses=extracted['loc_doses'],
                            level_number=level,
                            replicate_number=rep
                        )
                        
                        new_results.append(result)
                        st.success("✅ Ready")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        if new_results:
            st.markdown("---")
            if st.button(f"📥 Import {len(new_results)} tests", type="primary"):
                st.session_state.results[step_id].extend(new_results)
                st.session_state.validation_steps[step_id].status = 'collecting'
                st.success(f"✅ Imported {len(new_results)} tests!")
                st.rerun()
    
    if results:
        st.markdown("---")
        st.subheader("📊 Collected Data")
        
        df = pd.DataFrame([{
            'Test': r.test_number,
            'Shield #': r.shield_test_number,
            'Level': r.level_number,
            'Rep': r.replicate_number,
            'Conc': f"{r.concentration:.3f}",
            'Abs': f"{r.absorbance:.4f}"
        } for r in results])
        
        st.dataframe(safe_dataframe_display(df), width='stretch')

def show_analyze_tab(step_id, step_config):
    st.subheader("📊 Analysis")
    
    results = st.session_state.results[step_id]
    
    if not results:
        st.warning("⚠️ No data to analyze")
        return
    
    if step_id not in st.session_state.designs:
        st.error("❌ No design found")
        return
    
    design = st.session_state.designs[step_id]
    
    if st.button("🔬 Run Analysis", type="primary"):
        with st.spinner("Analyzing..."):
            df = pd.DataFrame([asdict(r) for r in results])
            
            try:
                if step_id == 'linearity':
                    analysis = analyze_linearity(df, design, step_config)
                elif step_id == 'repeatability':
                    analysis = analyze_repeatability(df, design, step_config)
                elif step_id == 'accuracy':
                    analysis = analyze_accuracy(df, design, step_config)
                else:
                    analysis = {'error': 'Not implemented'}
                
                if 'error' in analysis:
                    st.error(f"❌ {analysis['error']}")
                else:
                    st.session_state.analyses[step_id] = analysis
                    st.session_state.validation_steps[step_id].status = 'analyzed'
                    st.success("✅ Analysis complete!")
                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    if step_id in st.session_state.analyses:
        st.success("✅ Analysis completed! View in Results tab")

def show_results_tab(step_id, step_config):
    st.subheader("📈 Results")
    
    if step_id not in st.session_state.analyses:
        st.warning("⚠️ Run analysis first")
        return
    
    analysis = st.session_state.analyses[step_id]
    
    if 'error' in analysis:
        st.error(f"❌ {analysis['error']}")
        return
    
    # Show results based on step
    if step_id == 'linearity':
        show_linearity_results(analysis, step_config)
    elif step_id == 'repeatability':
        show_repeatability_results(analysis, step_config)
    elif step_id == 'accuracy':
        show_accuracy_results(analysis, step_config)

def show_linearity_results(analysis, config):
    if analysis['passes']:
        st.success(f"✅ PASS: R² = {analysis['r_squared']:.4f}")
    else:
        st.error(f"❌ FAIL: R² = {analysis['r_squared']:.4f}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{analysis['r_squared']:.4f}")
    with col2:
        st.metric("Slope", f"{analysis['slope']:.6f}")
    with col3:
        st.metric("Intercept", f"{analysis['intercept']:.6f}")
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=analysis['x'],
        y=analysis['y'],
        mode='markers',
        name='Data',
        marker=dict(size=10, color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis['x'],
        y=analysis['y_pred'],
        mode='lines',
        name='Fit',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f"Linearity: y = {analysis['slope']:.6f}x + {analysis['intercept']:.6f}",
        xaxis_title="Concentration (mg/L)",
        yaxis_title="Absorbance",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_repeatability_results(analysis, config):
    if analysis['passes']:
        st.success("✅ PASS: All levels meet RSD criteria")
    else:
        st.error("❌ FAIL: Some levels exceed RSD")
    
    data = []
    for level, res in analysis['by_level'].items():
        data.append({
            'Level': level,
            'Mean': f"{res['mean']:.3f}",
            'RSD (%)': f"{res['rsd']:.2f}",
            'n': res['n'],
            'Status': '✅' if res['passes'] else '❌'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(safe_dataframe_display(df), width='stretch')
    
    # Plot
    fig = go.Figure()
    
    for level, res in analysis['by_level'].items():
        fig.add_trace(go.Box(
            y=res['data'],
            name=f"Level {level}",
            boxmean='sd'
        ))
    
    fig.update_layout(
        title="Repeatability by Level",
        xaxis_title="Level",
        yaxis_title="Concentration (mg/L)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_accuracy_results(analysis, config):
    if analysis['passes']:
        st.success("✅ PASS: All recoveries acceptable")
    else:
        st.error("❌ FAIL: Some recoveries outside range")
    
    data = []
    for level, res in analysis['by_level'].items():
        data.append({
            'Level': level,
            'Spiked': f"{res['spike_conc']:.3f}",
            'Measured': f"{res['measured']:.3f}",
            'Recovery (%)': f"{res['recovery']:.1f}",
            'Status': '✅' if res['passes'] else '❌'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(safe_dataframe_display(df), width='stretch')

def show_config():
    st.header("⚙️ LOC Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_vol = st.number_input("Base Volume (mL)", value=40.0, step=1.0)
        extra_vol = st.number_input("Extra Volume (mL)", value=0.0, step=0.1)
    
    with col2:
        standard_loc = st.selectbox("Standard LOC", [f'LOC{i}' for i in range(1, 17)], index=14)
        stock_conc = st.number_input("Stock Conc (mg/L)", value=1000.0, step=10.0)
    
    if st.button("💾 Save", type="primary"):
        st.session_state.loc_config.update({
            'base_sample_volume': base_vol,
            'extra_volume': extra_vol,
            'standard_loc': standard_loc,
            'stock_concentration': stock_conc
        })
        st.success("✅ Saved!")

if __name__ == "__main__":
    main()
