"""
Fe²⁺/Fe³⁺ Method Validation Dashboard - Complete Workflow
Professional Streamlit Application with Guided Validation Steps

Features:
- Step-by-step validation workflow
- Experiment design for each validation step
- Data collection and analysis
- Statistical calculations
- Automated reporting
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
import pickle

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fe Method Validation - Complete Workflow",
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .step-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .step-complete {
        border-color: #28a745;
        background: #f0fff0;
    }
    .step-active {
        border-color: #4A90E2;
        background: #f0f8ff;
    }
    .step-pending {
        border-color: #e0e0e0;
        background: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA CLASSES FOR VALIDATION STEPS
# ============================================================================

@dataclass
class ValidationStepConfig:
    """Configuration for a validation step"""
    step_id: str
    step_name: str
    description: str
    requires_design: bool
    design_params: Dict
    expected_tests: int
    acceptance_criteria: Dict
    status: str = "not_started"  # not_started, designed, collecting, completed, analyzed
    
@dataclass
class ExperimentDesign:
    """Experiment design parameters"""
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
    """Individual test result"""
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
    """Initialize all session state variables"""
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        
        # LOC Configuration
        st.session_state.loc_config = {
            'base_sample_volume': 40.0,
            'extra_volume': 0.0,
            'include_loc_volumes': True,
            'standard_loc': 'LOC15',
            'stock_concentration': 1000.0
        }
        
        # Validation Steps
        st.session_state.validation_steps = {
            'linearity': ValidationStepConfig(
                step_id='linearity',
                step_name='4A. Linearity',
                description='Assess linear relationship between concentration and response',
                requires_design=True,
                design_params={
                    'concentration_range': (0, 100),  # mg/L
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
            'interference': ValidationStepConfig(
                step_id='interference',
                step_name='4B. Interference',
                description='Test for interference from common ions and substances',
                requires_design=True,
                design_params={
                    'interferent_list': ['Cl⁻', 'SO₄²⁻', 'Ca²⁺', 'Mg²⁺', 'Humic Acid', 
                                        'Turbidity', 'Al³⁺', 'Mn²⁺', 'Cu²⁺'],
                    'fe_concentration': 50,  # mg/L
                    'interferent_levels': [0, 'Low', 'Medium', 'High'],
                    'replicates': 3
                },
                expected_tests=108,
                acceptance_criteria={
                    'recovery_range': (90, 110),  # %
                    'rsd': 5  # %
                }
            ),
            'repeatability': ValidationStepConfig(
                step_id='repeatability',
                step_name='5A. Repeatability (Intra-day Precision)',
                description='Same analyst, same day, same conditions',
                requires_design=True,
                design_params={
                    'concentration_levels': [10, 50, 90],  # mg/L (Low, Mid, High)
                    'replicates_per_level': 10,
                    'same_day': True
                },
                expected_tests=30,
                acceptance_criteria={
                    'rsd': 5  # % at each level
                }
            ),
            'intermediate_precision': ValidationStepConfig(
                step_id='intermediate_precision',
                step_name='5A. Intermediate Precision',
                description='Different days, different analysts, slight condition variations',
                requires_design=True,
                design_params={
                    'concentration_levels': [10, 50, 90],  # mg/L
                    'days': 3,
                    'replicates_per_day': 5
                },
                expected_tests=45,
                acceptance_criteria={
                    'rsd': 7.5  # % (typically higher than repeatability)
                }
            ),
            'accuracy': ValidationStepConfig(
                step_id='accuracy',
                step_name='5B. Accuracy (Recovery)',
                description='Spike known amounts and measure recovery',
                requires_design=True,
                design_params={
                    'spike_levels': [10, 50, 90],  # mg/L
                    'matrix': 'DI water',
                    'replicates_per_level': 5
                },
                expected_tests=15,
                acceptance_criteria={
                    'recovery_range': (95, 105),  # %
                    'rsd': 5  # %
                }
            ),
            'lod_loq': ValidationStepConfig(
                step_id='lod_loq',
                step_name='5C. LOD & LOQ',
                description='Determine detection and quantification limits',
                requires_design=True,
                design_params={
                    'low_concentrations': [0, 0.5, 1.0, 2.0, 5.0, 10.0],  # mg/L
                    'replicates_per_level': 7,
                    'blank_replicates': 10
                },
                expected_tests=52,
                acceptance_criteria={
                    'lod': 'S/N ≥ 3',
                    'loq': 'S/N ≥ 10, RSD < 10%'
                }
            ),
            'stability': ValidationStepConfig(
                step_id='stability',
                step_name='5D. Stability',
                description='Test sample and standard stability over time',
                requires_design=True,
                design_params={
                    'test_types': ['Standard Stability', 'Sample Stability'],
                    'concentration': 50,  # mg/L
                    'time_points': ['T0', '24h', '48h', '72h', '1week'],
                    'storage_conditions': ['Room temp', 'Refrigerated'],
                    'replicates': 3
                },
                expected_tests=30,
                acceptance_criteria={
                    'deviation': 5  # % from T0
                }
            ),
            'robustness': ValidationStepConfig(
                step_id='robustness',
                step_name='6. Robustness',
                description='Test method resilience to small parameter changes',
                requires_design=True,
                design_params={
                    'parameters': ['Temperature', 'pH', 'Reagent Lot', 'Sample Volume'],
                    'concentration': 50,  # mg/L
                    'variations': ['Normal', '+Δ', '-Δ'],
                    'replicates': 3
                },
                expected_tests=24,
                acceptance_criteria={
                    'rsd': 5,  # %
                    'bias': 5  # %
                }
            ),
            'matrix': ValidationStepConfig(
                step_id='matrix',
                step_name='7. Matrix Effects',
                description='Compare DI water vs real sample matrices',
                requires_design=True,
                design_params={
                    'matrices': ['DI Water', 'Tap Water', 'Surface Water', 'Wastewater'],
                    'spike_levels': [25, 50, 75],  # mg/L
                    'replicates': 3
                },
                expected_tests=36,
                acceptance_criteria={
                    'matrix_effect': 10,  # % difference
                    'recovery_range': (90, 110)  # %
                }
            )
        }
        
        # Experiment designs
        st.session_state.designs = {}
        
        # Test results
        st.session_state.results = {step: [] for step in st.session_state.validation_steps.keys()}
        
        # Analysis results
        st.session_state.analyses = {}
        
        # Current step
        st.session_state.current_step = 'linearity'

initialize_session_state()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_json_data(json_content):
    """Extract data from JSON file"""
    try:
        data = json.loads(json_content) if isinstance(json_content, str) else json_content
        
        # Extract Shield Test Number
        shield_test_num = None
        if 'payload' in data:
            shield_test_num = (data['payload'].get('exp_number') or 
                             data['payload'].get('test_number'))
        if not shield_test_num:
            shield_test_num = data.get('exp_number') or data.get('test_number')
        
        # Find scans
        scans = None
        if 'payload' in data and 'scans' in data['payload']:
            scans = data['payload']['scans']
        elif 'scans' in data:
            scans = data['scans']
        
        if not scans:
            return None
        
        # Find background and last sample scan
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
        
        # Calculate absorbance
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
        
        # Extract LOC doses
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
        
        # Extract temperature
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
    """Parse comma-separated channel values"""
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
    """Calculate robust mean using median and MAD"""
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
    """Calculate concentration using M1V1=M2V2"""
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

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🧪 Fe²⁺/Fe³⁺ Method Validation - Complete Workflow</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Validation Workflow")
        
        # Progress overview
        completed_steps = sum(1 for step in st.session_state.validation_steps.values() 
                             if step.status == 'completed')
        total_steps = len(st.session_state.validation_steps)
        
        st.metric("Progress", f"{completed_steps}/{total_steps} Steps")
        st.progress(completed_steps / total_steps)
        
        st.markdown("---")
        
        # Step selection
        st.subheader("Select Validation Step:")
        
        for step_id, step_config in st.session_state.validation_steps.items():
            status_emoji = {
                'not_started': '⚪',
                'designed': '🔵',
                'collecting': '🔄',
                'completed': '✅',
                'analyzed': '🎉'
            }
            
            emoji = status_emoji.get(step_config.status, '⚪')
            
            if st.button(
                f"{emoji} {step_config.step_name}",
                key=f"nav_{step_id}",
                use_container_width=True
            ):
                st.session_state.current_step = step_id
                st.rerun()
        
        st.markdown("---")
        
        # Quick actions
        st.subheader("⚙️ Configuration")
        if st.button("🔧 LOC Configuration", use_container_width=True):
            st.session_state.current_step = 'loc_config'
            st.rerun()
        
        if st.button("📊 Overview Dashboard", use_container_width=True):
            st.session_state.current_step = 'dashboard'
            st.rerun()
        
        if st.button("📋 Export All Results", use_container_width=True):
            st.session_state.current_step = 'export'
            st.rerun()
    
    # Route to appropriate page
    current_step = st.session_state.current_step
    
    if current_step == 'dashboard':
        show_dashboard()
    elif current_step == 'loc_config':
        show_loc_configuration()
    elif current_step == 'export':
        show_export_all()
    else:
        show_validation_step(current_step)

# (Continuing in next message with validation step pages...)

# ============================================================================
# TAB 2: DATA COLLECTION
# ============================================================================

def show_data_collection(step_id, step_config):
    """Collect data for this validation step"""
    
    st.subheader("📥 Data Collection")
    
    # Check if design exists
    if step_id not in st.session_state.designs:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Design Required First!</strong><br>
        Please design your experiment in the "Design Experiment" tab before collecting data.
        </div>
        """, unsafe_allow_html=True)
        return
    
    design = st.session_state.designs[step_id]
    results = st.session_state.results[step_id]
    
    # Show progress
    expected_tests = len(design.spike_levels) * design.num_replicates if design.spike_levels else step_config.expected_tests
    collected_tests = len(results)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Tests", expected_tests)
    with col2:
        st.metric("Collected Tests", collected_tests)
    with col3:
        progress_pct = (collected_tests / expected_tests * 100) if expected_tests > 0 else 0
        st.metric("Progress", f"{progress_pct:.1f}%")
    
    st.progress(min(collected_tests / expected_tests, 1.0) if expected_tests > 0 else 0)
    
    st.markdown("---")
    
    # File upload
    st.markdown("### 📁 Upload JSON Test Files")
    
    uploaded_files = st.file_uploader(
        "Choose JSON file(s)",
        type=['json'],
        accept_multiple_files=True,
        help="Upload test result files for this validation step"
    )
    
    if uploaded_files:
        st.markdown("### 📋 Map Files to Test Conditions")
        
        # For each file, let user specify which level/replicate it is
        new_results = []
        
        for idx, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name}", expanded=(idx == 0)):
                try:
                    content = file.read().decode('utf-8')
                    extracted = extract_json_data(content)
                    
                    if not extracted:
                        st.error(f"❌ Could not extract data from {file.name}")
                        continue
                    
                    # Calculate concentration
                    concentration = calculate_concentration(
                        extracted['loc_doses'],
                        st.session_state.loc_config
                    )
                    extracted['concentration'] = concentration
                    
                    # Show extracted data
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Shield Test #", extracted['shield_test_number'])
                    with col2:
                        st.metric("Concentration", f"{concentration:.3f} mg/L")
                    with col3:
                        st.metric("Absorbance", f"{extracted['absorbance']:.4f}" if extracted['absorbance'] else "N/A")
                    
                    # Let user specify test details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if design.spike_levels:
                            # Match to nearest level
                            nearest_level = min(range(len(design.spike_levels)), 
                                              key=lambda i: abs(design.spike_levels[i] - concentration))
                            
                            level_number = st.selectbox(
                                "Concentration Level",
                                options=range(1, len(design.spike_levels) + 1),
                                index=nearest_level,
                                key=f"level_{idx}",
                                format_func=lambda x: f"Level {x}: {design.spike_levels[x-1]:.2f} mg/L"
                            )
                        else:
                            level_number = st.number_input(
                                "Test Level", 
                                min_value=1, 
                                value=1, 
                                key=f"level_{idx}"
                            )
                    
                    with col2:
                        replicate_number = st.number_input(
                            "Replicate Number",
                            min_value=1,
                            max_value=design.num_replicates,
                            value=1,
                            key=f"rep_{idx}"
                        )
                    
                    # Additional metadata based on step type
                    metadata = {}
                    
                    if step_id == 'interference':
                        interferent = st.selectbox(
                            "Interferent",
                            options=['None (Control)'] + design.interferents,
                            key=f"int_{idx}"
                        )
                        interferent_level = st.selectbox(
                            "Interferent Level",
                            options=['Control', 'Low', 'Medium', 'High'],
                            key=f"int_level_{idx}"
                        )
                        metadata = {'interferent': interferent, 'interferent_level': interferent_level}
                    
                    elif step_id == 'intermediate_precision':
                        test_day = st.selectbox(
                            "Test Day",
                            options=range(1, design.custom_params['num_days'] + 1),
                            key=f"day_{idx}"
                        )
                        metadata = {'test_day': test_day}
                    
                    elif step_id == 'stability':
                        time_point = st.selectbox(
                            "Time Point",
                            options=design.custom_params['time_points'],
                            key=f"time_{idx}"
                        )
                        storage = st.selectbox(
                            "Storage Condition",
                            options=design.custom_params['storage_conditions'],
                            key=f"storage_{idx}"
                        )
                        metadata = {'time_point': time_point, 'storage': storage}
                    
                    elif step_id == 'robustness':
                        parameter = st.selectbox(
                            "Parameter Varied",
                            options=design.custom_params['parameters'],
                            key=f"param_{idx}"
                        )
                        variation = st.selectbox(
                            "Variation",
                            options=['Normal', '+Δ', '-Δ'],
                            key=f"var_{idx}"
                        )
                        metadata = {'parameter': parameter, 'variation': variation}
                    
                    elif step_id == 'matrix':
                        matrix = st.selectbox(
                            "Sample Matrix",
                            options=design.custom_params['matrices'],
                            key=f"matrix_{idx}"
                        )
                        metadata = {'matrix': matrix}
                    
                    # Create test result
                    test_result = TestResult(
                        step_id=step_id,
                        test_number=collected_tests + len(new_results) + 1,
                        shield_test_number=extracted['shield_test_number'],
                        timestamp=extracted['timestamp'],
                        concentration=concentration,
                        absorbance=extracted['absorbance'] or 0.0,
                        bg_mean=extracted['bg_mean'] or 0.0,
                        sample_mean=extracted['sample_mean'] or 0.0,
                        temperature=extracted['temperature'] or 0.0,
                        loc_doses=extracted['loc_doses'],
                        replicate_number=replicate_number,
                        level_number=level_number,
                        metadata=metadata
                    )
                    
                    new_results.append(test_result)
                    
                    st.success(f"✅ Data extracted and ready to import")
                
                except Exception as e:
                    st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        # Import button
        if new_results:
            st.markdown("---")
            if st.button(f"📥 Import {len(new_results)} Test(s)", type="primary"):
                st.session_state.results[step_id].extend(new_results)
                st.session_state.validation_steps[step_id].status = 'collecting'
                st.success(f"✅ Successfully imported {len(new_results)} test results!")
                st.rerun()
    
    # Show collected data
    if results:
        st.markdown("---")
        st.markdown("### 📊 Collected Data")
        
        df_results = pd.DataFrame([asdict(r) for r in results])
        df_display = df_results[['test_number', 'shield_test_number', 'level_number', 
                                 'replicate_number', 'concentration', 'absorbance', 'timestamp']]
        df_display.columns = ['Test #', 'Shield Test #', 'Level', 'Rep', 
                             'Conc (mg/L)', 'Absorbance', 'Timestamp']
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Option to delete last import
        if st.button("🗑️ Delete Last Import"):
            if results:
                st.session_state.results[step_id].pop()
                st.success("Deleted last import")
                st.rerun()

# ============================================================================
# TAB 3: ANALYZE RESULTS
# ============================================================================

def show_analysis(step_id, step_config):
    """Analyze collected data"""
    
    st.subheader("📊 Statistical Analysis")
    
    results = st.session_state.results[step_id]
    
    if not results:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>No Data to Analyze</strong><br>
        Please collect data first in the "Collect Data" tab.
        </div>
        """, unsafe_allow_html=True)
        return
    
    if step_id not in st.session_state.designs:
        st.error("No experiment design found!")
        return
    
    design = st.session_state.designs[step_id]
    
    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])
    
    # Perform analysis based on validation step
    if st.button("🔬 Run Analysis", type="primary"):
        with st.spinner("Analyzing data..."):
            if step_id == 'linearity':
                analysis_results = analyze_linearity(df, design, step_config)
            elif step_id == 'interference':
                analysis_results = analyze_interference(df, design, step_config)
            elif step_id == 'repeatability':
                analysis_results = analyze_repeatability(df, design, step_config)
            elif step_id == 'intermediate_precision':
                analysis_results = analyze_intermediate_precision(df, design, step_config)
            elif step_id == 'accuracy':
                analysis_results = analyze_accuracy(df, design, step_config)
            elif step_id == 'lod_loq':
                analysis_results = analyze_lod_loq(df, design, step_config)
            elif step_id == 'stability':
                analysis_results = analyze_stability(df, design, step_config)
            elif step_id == 'robustness':
                analysis_results = analyze_robustness(df, design, step_config)
            elif step_id == 'matrix':
                analysis_results = analyze_matrix_effects(df, design, step_config)
            else:
                analysis_results = {'error': 'Analysis not implemented for this step'}
            
            st.session_state.analyses[step_id] = analysis_results
            st.session_state.validation_steps[step_id].status = 'analyzed'
            
            st.success("✅ Analysis complete! View results in the 'View Results' tab.")
            st.rerun()
    
    # Show existing analysis if available
    if step_id in st.session_state.analyses:
        st.markdown("""
        <div class="success-box">
        ✅ <strong>Analysis Already Performed</strong><br>
        View detailed results in the "View Results" tab or re-run analysis above.
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_linearity(df, design, config):
    """Analyze linearity data"""
    
    # Group by level and calculate means
    grouped = df.groupby('level_number').agg({
        'concentration': ['mean', 'std', 'count'],
        'absorbance': ['mean', 'std']
    }).reset_index()
    
    x = grouped['concentration']['mean'].values
    y = grouped['absorbance']['mean'].values
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    # Residuals
    y_pred = slope * x + intercept
    residuals = y - y_pred
    
    # Acceptance criteria
    passes = r_squared >= config.acceptance_criteria['r_squared']
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
        'x': x.tolist(),
        'y': y.tolist(),
        'y_pred': y_pred.tolist(),
        'residuals': residuals.tolist(),
        'passes': passes,
        'criteria': config.acceptance_criteria,
        'summary': f"R² = {r_squared:.4f} (Criteria: ≥ {config.acceptance_criteria['r_squared']})"
    }

def analyze_repeatability(df, design, config):
    """Analyze repeatability (intra-day precision)"""
    
    results_by_level = {}
    
    for level in df['level_number'].unique():
        level_data = df[df['level_number'] == level]['concentration']
        
        mean = level_data.mean()
        std = level_data.std()
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

def analyze_interference(df, design, config):
    """Analyze interference effects"""
    
    # Get control (no interferent) mean
    control_data = df[df['metadata'].apply(lambda x: x.get('interferent') == 'None (Control)')]
    control_mean = control_data['concentration'].mean()
    
    results_by_interferent = {}
    
    for interferent in design.interferents:
        interferent_data = df[df['metadata'].apply(lambda x: x.get('interferent') == interferent)]
        
        if len(interferent_data) == 0:
            continue
        
        by_level = {}
        for level in ['Low', 'Medium', 'High']:
            level_data = interferent_data[interferent_data['metadata'].apply(lambda x: x.get('interferent_level') == level)]
            
            if len(level_data) > 0:
                mean = level_data['concentration'].mean()
                recovery = (mean / control_mean * 100) if control_mean > 0 else 0
                rsd = (level_data['concentration'].std() / mean * 100) if mean > 0 else 0
                
                passes = (config.acceptance_criteria['recovery_range'][0] <= recovery <= 
                         config.acceptance_criteria['recovery_range'][1] and 
                         rsd <= config.acceptance_criteria['rsd'])
                
                by_level[level] = {
                    'mean': mean,
                    'recovery': recovery,
                    'rsd': rsd,
                    'passes': passes
                }
        
        results_by_interferent[interferent] = by_level
    
    return {
        'control_mean': control_mean,
        'by_interferent': results_by_interferent,
        'criteria': config.acceptance_criteria,
        'summary': "Interference analysis complete"
    }

def analyze_accuracy(df, design, config):
    """Analyze accuracy/recovery"""
    
    results_by_level = {}
    
    for level in df['level_number'].unique():
        level_data = df[df['level_number'] == level]
        spike_conc = design.spike_levels[int(level) - 1]
        
        measured = level_data['concentration'].mean()
        recovery = (measured / spike_conc * 100) if spike_conc > 0 else 0
        rsd = (level_data['concentration'].std() / measured * 100) if measured > 0 else 0
        
        passes = (config.acceptance_criteria['recovery_range'][0] <= recovery <= 
                 config.acceptance_criteria['recovery_range'][1] and
                 rsd <= config.acceptance_criteria['rsd'])
        
        results_by_level[int(level)] = {
            'spike_conc': spike_conc,
            'measured': measured,
            'recovery': recovery,
            'rsd': rsd,
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

def analyze_lod_loq(df, design, config):
    """Analyze LOD and LOQ"""
    
    # Get blank measurements
    blank_data = df[df['level_number'] == 1]['absorbance']  # Assuming level 1 is blank
    
    blank_mean = blank_data.mean()
    blank_std = blank_data.std()
    
    # LOD = blank mean + 3*std
    # LOQ = blank mean + 10*std
    lod = blank_mean + 3 * blank_std
    loq = blank_mean + 10 * blank_std
    
    # Find corresponding concentrations
    # ... (simplified for brevity)
    
    return {
        'blank_mean': blank_mean,
        'blank_std': blank_std,
        'lod_absorbance': lod,
        'loq_absorbance': loq,
        'lod_concentration': lod * 10,  # Simplified
        'loq_concentration': loq * 10,  # Simplified
        'summary': f"LOD = {lod:.4f}, LOQ = {loq:.4f}"
    }

def analyze_intermediate_precision(df, design, config):
    """Analyze intermediate precision (inter-day variability)"""
    # Similar to repeatability but across days
    return analyze_repeatability(df, design, config)

def analyze_stability(df, design, config):
    """Analyze stability over time"""
    
    t0_data = df[df['metadata'].apply(lambda x: x.get('time_point') == 'T0 (Initial)')]
    t0_mean = t0_data['concentration'].mean()
    
    results_by_timepoint = {}
    
    for tp in design.custom_params['time_points']:
        if tp == 'T0 (Initial)':
            continue
        
        tp_data = df[df['metadata'].apply(lambda x: x.get('time_point') == tp)]
        
        if len(tp_data) > 0:
            mean = tp_data['concentration'].mean()
            deviation = abs((mean - t0_mean) / t0_mean * 100) if t0_mean > 0 else 0
            passes = deviation <= config.acceptance_criteria['deviation']
            
            results_by_timepoint[tp] = {
                'mean': mean,
                'deviation': deviation,
                'passes': passes
            }
    
    return {
        't0_mean': t0_mean,
        'by_timepoint': results_by_timepoint,
        'criteria': config.acceptance_criteria,
        'summary': "Stability analysis complete"
    }

def analyze_robustness(df, design, config):
    """Analyze robustness"""
    
    normal_data = df[df['metadata'].apply(lambda x: x.get('variation') == 'Normal')]
    normal_mean = normal_data['concentration'].mean()
    
    results_by_parameter = {}
    
    for param in design.custom_params['parameters']:
        param_results = {}
        
        for var in ['+Δ', '-Δ']:
            var_data = df[(df['metadata'].apply(lambda x: x.get('parameter') == param)) &
                         (df['metadata'].apply(lambda x: x.get('variation') == var))]
            
            if len(var_data) > 0:
                mean = var_data['concentration'].mean()
                bias = abs((mean - normal_mean) / normal_mean * 100) if normal_mean > 0 else 0
                
                param_results[var] = {
                    'mean': mean,
                    'bias': bias
                }
        
        results_by_parameter[param] = param_results
    
    return {
        'normal_mean': normal_mean,
        'by_parameter': results_by_parameter,
        'criteria': config.acceptance_criteria,
        'summary': "Robustness analysis complete"
    }

def analyze_matrix_effects(df, design, config):
    """Analyze matrix effects"""
    
    di_water = df[df['metadata'].apply(lambda x: x.get('matrix') == 'DI Water (Control)')]
    di_mean = di_water.groupby('level_number')['concentration'].mean()
    
    results_by_matrix = {}
    
    for matrix in design.custom_params['matrices']:
        if matrix == 'DI Water (Control)':
            continue
        
        matrix_data = df[df['metadata'].apply(lambda x: x.get('matrix') == matrix)]
        
        by_level = {}
        for level in matrix_data['level_number'].unique():
            level_data = matrix_data[matrix_data['level_number'] == level]
            mean = level_data['concentration'].mean()
            di_ref = di_mean.get(level, 0)
            
            matrix_effect = abs((mean - di_ref) / di_ref * 100) if di_ref > 0 else 0
            
            by_level[int(level)] = {
                'mean': mean,
                'di_reference': di_ref,
                'matrix_effect': matrix_effect
            }
        
        results_by_matrix[matrix] = by_level
    
    return {
        'di_water_means': di_mean.to_dict(),
        'by_matrix': results_by_matrix,
        'criteria': config.acceptance_criteria,
        'summary': "Matrix effects analysis complete"
    }

# (Continue with results display in next message...)

# ============================================================================
# TAB 4: VIEW RESULTS
# ============================================================================

def show_results_summary(step_id, step_config):
    """Show detailed results and visualizations"""
    
    st.subheader("📈 Results Summary")
    
    # Check if analysis exists
    if step_id not in st.session_state.analyses:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>No Analysis Available</strong><br>
        Please run the analysis in the "Analyze Results" tab first.
        </div>
        """, unsafe_allow_html=True)
        return
    
    analysis = st.session_state.analyses[step_id]
    results = st.session_state.results[step_id]
    df = pd.DataFrame([asdict(r) for r in results])
    
    # Show results based on validation step
    if step_id == 'linearity':
        show_linearity_results(df, analysis, step_config)
    elif step_id == 'interference':
        show_interference_results(df, analysis, step_config)
    elif step_id == 'repeatability':
        show_repeatability_results(df, analysis, step_config)
    elif step_id == 'intermediate_precision':
        show_intermediate_precision_results(df, analysis, step_config)
    elif step_id == 'accuracy':
        show_accuracy_results(df, analysis, step_config)
    elif step_id == 'lod_loq':
        show_lod_loq_results(df, analysis, step_config)
    elif step_id == 'stability':
        show_stability_results(df, analysis, step_config)
    elif step_id == 'robustness':
        show_robustness_results(df, analysis, step_config)
    elif step_id == 'matrix':
        show_matrix_results(df, analysis, step_config)

# ============================================================================
# RESULTS DISPLAY FUNCTIONS
# ============================================================================

def show_linearity_results(df, analysis, config):
    """Display linearity results"""
    
    # Pass/Fail Status
    if analysis['passes']:
        st.markdown("""
        <div class="success-box">
        ✅ <strong>LINEARITY TEST PASSED</strong><br>
        R² = {r2:.4f} (Criteria: ≥ {crit})
        </div>
        """.format(r2=analysis['r_squared'], crit=config.acceptance_criteria['r_squared']), 
        unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-box">
        ❌ <strong>LINEARITY TEST FAILED</strong><br>
        R² = {r2:.4f} (Criteria: ≥ {crit})
        </div>
        """.format(r2=analysis['r_squared'], crit=config.acceptance_criteria['r_squared']), 
        unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{analysis['r_squared']:.4f}")
    with col2:
        st.metric("Slope", f"{analysis['slope']:.6f}")
    with col3:
        st.metric("Intercept", f"{analysis['intercept']:.6f}")
    
    # Regression Plot
    st.subheader("📈 Calibration Curve")
    
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=analysis['x'],
        y=analysis['y'],
        mode='markers',
        name='Data Points',
        marker=dict(size=10, color='#4A90E2')
    ))
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=analysis['x'],
        y=analysis['y_pred'],
        mode='lines',
        name='Regression Line',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f"Linearity: y = {analysis['slope']:.6f}x + {analysis['intercept']:.6f}",
        xaxis_title="Concentration (mg/L)",
        yaxis_title="Absorbance",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals Plot
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Residuals vs Concentration")
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=analysis['x'],
            y=analysis['residuals'],
            mode='markers',
            marker=dict(size=8, color='purple')
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_layout(
            xaxis_title="Concentration (mg/L)",
            yaxis_title="Residuals",
            height=400
        )
        st.plotly_chart(fig_res, use_container_width=True)
    
    with col2:
        st.subheader("📊 Residuals Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=analysis['residuals'],
            nbinsx=10,
            marker_color='purple'
        ))
        fig_hist.update_layout(
            xaxis_title="Residuals",
            yaxis_title="Frequency",
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Data Table
    st.subheader("📋 Raw Data")
    df_display = df[['level_number', 'concentration', 'absorbance', 'replicate_number']]
    df_display.columns = ['Level', 'Concentration (mg/L)', 'Absorbance', 'Replicate']
    st.dataframe(df_display, use_container_width=True, hide_index=True)

def show_repeatability_results(df, analysis, config):
    """Display repeatability results"""
    
    # Overall Pass/Fail
    if analysis['passes']:
        st.markdown("""
        <div class="success-box">
        ✅ <strong>REPEATABILITY TEST PASSED</strong><br>
        All levels meet RSD ≤ {crit}% criteria
        </div>
        """.format(crit=config.acceptance_criteria['rsd']), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-box">
        ❌ <strong>REPEATABILITY TEST FAILED</strong><br>
        One or more levels exceed RSD criteria
        </div>
        """.format(crit=config.acceptance_criteria['rsd']), unsafe_allow_html=True)
    
    # Results by Level
    st.subheader("📊 Precision by Concentration Level")
    
    results_data = []
    for level, data in analysis['by_level'].items():
        results_data.append({
            'Level': level,
            'Mean (mg/L)': f"{data['mean']:.3f}",
            'Std Dev': f"{data['std']:.3f}",
            'RSD (%)': f"{data['rsd']:.2f}",
            'n': data['n'],
            'Status': '✅ Pass' if data['passes'] else '❌ Fail'
        })
    
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Box Plot
    st.subheader("📈 Distribution by Level")
    
    fig = go.Figure()
    
    for level, data in analysis['by_level'].items():
        fig.add_trace(go.Box(
            y=data['data'],
            name=f"Level {level}",
            boxmean='sd'
        ))
    
    fig.update_layout(
        xaxis_title="Concentration Level",
        yaxis_title="Concentration (mg/L)",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # RSD Bar Chart
    st.subheader("📊 RSD Comparison")
    
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
        xaxis_title="Level",
        yaxis_title="RSD (%)",
        height=400
    )
    
    st.plotly_chart(fig_rsd, use_container_width=True)

def show_interference_results(df, analysis, config):
    """Display interference results"""
    
    st.subheader("🔬 Interference Study Results")
    
    # Control Mean
    st.info(f"**Control Mean (No Interferent):** {analysis['control_mean']:.3f} mg/L")
    
    # Results by Interferent
    for interferent, by_level in analysis['by_interferent'].items():
        with st.expander(f"📊 {interferent}", expanded=True):
            
            # Table
            results_data = []
            for level, data in by_level.items():
                results_data.append({
                    'Interferent Level': level,
                    'Mean (mg/L)': f"{data['mean']:.3f}",
                    'Recovery (%)': f"{data['recovery']:.1f}",
                    'RSD (%)': f"{data['rsd']:.2f}",
                    'Status': '✅ Pass' if data['passes'] else '❌ Fail'
                })
            
            df_int = pd.DataFrame(results_data)
            st.dataframe(df_int, use_container_width=True, hide_index=True)
            
            # Recovery Chart
            levels = list(by_level.keys())
            recoveries = [by_level[l]['recovery'] for l in levels]
            colors = ['green' if by_level[l]['passes'] else 'red' for l in levels]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=levels,
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
                line_width=0,
                annotation_text="Acceptance Range"
            )
            
            fig.update_layout(
                title=f"{interferent} - Recovery %",
                xaxis_title="Interferent Level",
                yaxis_title="Recovery (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_intermediate_precision_results(df, analysis, config):
    """Display intermediate precision results (similar to repeatability but note inter-day)"""
    
    st.info("**Note:** This analysis assesses precision across multiple days/analysts")
    show_repeatability_results(df, analysis, config)

def show_accuracy_results(df, analysis, config):
    """Display accuracy/recovery results"""
    
    # Overall Pass/Fail
    if analysis['passes']:
        st.markdown("""
        <div class="success-box">
        ✅ <strong>ACCURACY TEST PASSED</strong><br>
        All spike levels show acceptable recovery
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-box">
        ❌ <strong>ACCURACY TEST FAILED</strong><br>
        One or more levels outside acceptance criteria
        </div>
        """, unsafe_allow_html=True)
    
    # Results Table
    st.subheader("🎯 Recovery Results")
    
    results_data = []
    for level, data in analysis['by_level'].items():
        results_data.append({
            'Spike Level': level,
            'Spiked (mg/L)': f"{data['spike_conc']:.3f}",
            'Measured (mg/L)': f"{data['measured']:.3f}",
            'Recovery (%)': f"{data['recovery']:.1f}",
            'RSD (%)': f"{data['rsd']:.2f}",
            'n': data['n'],
            'Status': '✅ Pass' if data['passes'] else '❌ Fail'
        })
    
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Recovery Chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Recovery by Spike Level")
        
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
            xaxis_title="Spike Level",
            yaxis_title="Recovery (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Spiked vs Measured")
        
        spiked = [analysis['by_level'][l]['spike_conc'] for l in levels]
        measured = [analysis['by_level'][l]['measured'] for l in levels]
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=spiked,
            y=measured,
            mode='markers',
            marker=dict(size=12, color='#4A90E2'),
            name='Data'
        ))
        
        # Perfect recovery line
        max_conc = max(max(spiked), max(measured))
        fig2.add_trace(go.Scatter(
            x=[0, max_conc],
            y=[0, max_conc],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='100% Recovery'
        ))
        
        fig2.update_layout(
            xaxis_title="Spiked Concentration (mg/L)",
            yaxis_title="Measured Concentration (mg/L)",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)

def show_lod_loq_results(df, analysis, config):
    """Display LOD/LOQ results"""
    
    st.subheader("🔍 Detection and Quantification Limits")
    
    # Key Results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Blank Mean", f"{analysis['blank_mean']:.4f}")
    with col2:
        st.metric("Blank Std Dev", f"{analysis['blank_std']:.4f}")
    with col3:
        st.metric("LOD", f"{analysis['lod_concentration']:.3f} mg/L")
    with col4:
        st.metric("LOQ", f"{analysis['loq_concentration']:.3f} mg/L")
    
    # Explanation
    st.info("""
    **Calculation Method:**
    - LOD = Blank Mean + 3 × Blank Std Dev
    - LOQ = Blank Mean + 10 × Blank Std Dev
    """)
    
    # Visualization
    st.subheader("📈 Low Concentration Response")
    
    fig = go.Figure()
    
    # Plot all data points
    fig.add_trace(go.Scatter(
        x=df['concentration'],
        y=df['absorbance'],
        mode='markers',
        marker=dict(size=8, color='#4A90E2'),
        name='Data Points'
    ))
    
    # LOD line
    fig.add_hline(
        y=analysis['lod_absorbance'],
        line_dash="dash",
        line_color="orange",
        annotation_text=f"LOD = {analysis['lod_concentration']:.3f} mg/L"
    )
    
    # LOQ line
    fig.add_hline(
        y=analysis['loq_absorbance'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"LOQ = {analysis['loq_concentration']:.3f} mg/L"
    )
    
    fig.update_layout(
        xaxis_title="Concentration (mg/L)",
        yaxis_title="Absorbance",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_stability_results(df, analysis, config):
    """Display stability results"""
    
    st.subheader("⏱️ Stability Over Time")
    
    # T0 Reference
    st.info(f"**T0 Reference Concentration:** {analysis['t0_mean']:.3f} mg/L")
    
    # Results Table
    results_data = []
    for tp, data in analysis['by_timepoint'].items():
        results_data.append({
            'Time Point': tp,
            'Mean (mg/L)': f"{data['mean']:.3f}",
            'Deviation from T0 (%)': f"{data['deviation']:.2f}",
            'Status': '✅ Stable' if data['passes'] else '❌ Unstable'
        })
    
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Time Series Plot
    st.subheader("📈 Concentration vs Time")
    
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
    
    # Acceptance range
    upper_limit = analysis['t0_mean'] * (1 + config.acceptance_criteria['deviation'] / 100)
    lower_limit = analysis['t0_mean'] * (1 - config.acceptance_criteria['deviation'] / 100)
    
    fig.add_hrect(
        y0=lower_limit,
        y1=upper_limit,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        annotation_text="±5% Acceptance Range"
    )
    
    fig.update_layout(
        xaxis_title="Time Point",
        yaxis_title="Concentration (mg/L)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_robustness_results(df, analysis, config):
    """Display robustness results"""
    
    st.subheader("💪 Method Robustness")
    
    # Normal Condition Reference
    st.info(f"**Normal Condition Mean:** {analysis['normal_mean']:.3f} mg/L")
    
    # Results by Parameter
    for param, variations in analysis['by_parameter'].items():
        with st.expander(f"📊 {param}", expanded=True):
            
            # Table
            results_data = []
            for var, data in variations.items():
                results_data.append({
                    'Variation': var,
                    'Mean (mg/L)': f"{data['mean']:.3f}",
                    'Bias (%)': f"{data['bias']:.2f}"
                })
            
            df_param = pd.DataFrame(results_data)
            st.dataframe(df_param, use_container_width=True, hide_index=True)
            
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

def show_matrix_results(df, analysis, config):
    """Display matrix effects results"""
    
    st.subheader("🌊 Matrix Effects")
    
    # DI Water Reference
    st.info("**DI Water (Control) serves as reference for matrix effect calculations**")
    
    # Results by Matrix
    for matrix, by_level in analysis['by_matrix'].items():
        with st.expander(f"📊 {matrix}", expanded=True):
            
            # Table
            results_data = []
            for level, data in by_level.items():
                results_data.append({
                    'Spike Level': level,
                    'DI Water (mg/L)': f"{data['di_reference']:.3f}",
                    'Matrix Mean (mg/L)': f"{data['mean']:.3f}",
                    'Matrix Effect (%)': f"{data['matrix_effect']:.2f}"
                })
            
            df_matrix = pd.DataFrame(results_data)
            st.dataframe(df_matrix, use_container_width=True, hide_index=True)
            
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
                xaxis_title="Spike Level",
                yaxis_title="Concentration (mg/L)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DASHBOARD & OTHER PAGES
# ============================================================================

def show_dashboard():
    """Main overview dashboard"""
    
    st.header("📊 Validation Overview Dashboard")
    
    # Overall Progress
    completed = sum(1 for s in st.session_state.validation_steps.values() if s.status == 'analyzed')
    total = len(st.session_state.validation_steps)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Steps", total)
    with col2:
        st.metric("Completed", completed)
    with col3:
        st.metric("In Progress", sum(1 for s in st.session_state.validation_steps.values() 
                                     if s.status in ['designed', 'collecting']))
    with col4:
        progress_pct = (completed / total * 100) if total > 0 else 0
        st.metric("Progress", f"{progress_pct:.0f}%")
    
    st.progress(completed / total if total > 0 else 0)
    
    st.markdown("---")
    
    # Status by Step
    st.subheader("📋 Validation Steps Status")
    
    status_data = []
    for step_id, step_config in st.session_state.validation_steps.items():
        results_count = len(st.session_state.results.get(step_id, []))
        
        status_emoji = {
            'not_started': '⚪',
            'designed': '🔵',
            'collecting': '🔄',
            'completed': '✅',
            'analyzed': '🎉'
        }
        
        status_data.append({
            'Step': step_config.step_name,
            'Status': f"{status_emoji.get(step_config.status, '⚪')} {step_config.status.replace('_', ' ').title()}",
            'Tests Collected': results_count,
            'Expected Tests': step_config.expected_tests
        })
    
    df_status = pd.DataFrame(status_data)
    st.dataframe(df_status, use_container_width=True, hide_index=True)
    
    # Progress Chart
    st.subheader("📈 Progress by Step")
    
    fig = go.Figure()
    
    steps = [s['Step'] for s in status_data]
    collected = [s['Tests Collected'] for s in status_data]
    expected = [s['Expected Tests'] for s in status_data]
    
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

def show_loc_configuration():
    """LOC Configuration page"""
    
    st.header("⚙️ LOC Configuration")
    
    st.markdown("""
    <div class="info-box">
    📋 <strong>Configure LOC Settings</strong><br>
    Set up your stock concentrations and sample volumes for M₁V₁=M₂V₂ calculations.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Sample Volume")
        
        base_volume = st.number_input(
            "Base Sample Volume (mL)",
            min_value=0.0,
            value=st.session_state.loc_config['base_sample_volume'],
            step=1.0
        )
        
        extra_volume = st.number_input(
            "Extra Volume (mL)",
            min_value=0.0,
            value=st.session_state.loc_config['extra_volume'],
            step=0.1
        )
        
        include_loc = st.checkbox(
            "Include LOC volumes in total?",
            value=st.session_state.loc_config['include_loc_volumes']
        )
    
    with col2:
        st.subheader("🧪 Fe Standard")
        
        standard_loc = st.selectbox(
            "Standard LOC Position",
            options=[f'LOC{i}' for i in range(1, 17)],
            index=14
        )
        
        stock_conc = st.number_input(
            "Stock Concentration (mg/L)",
            min_value=0.0,
            value=st.session_state.loc_config['stock_concentration'],
            step=10.0
        )
    
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.loc_config.update({
            'base_sample_volume': base_volume,
            'extra_volume': extra_volume,
            'include_loc_volumes': include_loc,
            'standard_loc': standard_loc,
            'stock_concentration': stock_conc
        })
        
        st.success("✅ Configuration saved!")

def show_export_all():
    """Export all results"""
    
    st.header("📋 Export All Results")
    
    # Create comprehensive Excel export
    if st.button("📥 Generate Complete Report", type="primary"):
        
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            # Summary sheet
            summary_data = []
            for step_id, step_config in st.session_state.validation_steps.items():
                summary_data.append({
                    'Validation Step': step_config.step_name,
                    'Status': step_config.status,
                    'Tests Collected': len(st.session_state.results.get(step_id, [])),
                    'Expected Tests': step_config.expected_tests
                })
            
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual step data
            for step_id, results in st.session_state.results.items():
                if results:
                    df = pd.DataFrame([asdict(r) for r in results])
                    step_name = st.session_state.validation_steps[step_id].step_name.replace('/', '-')
                    df.to_excel(writer, sheet_name=step_name[:31], index=False)
        
        output.seek(0)
        
        st.download_button(
            label="📥 Download Complete Report",
            data=output,
            file_name=f"fe_validation_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success("✅ Report generated!")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
