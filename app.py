"""
KETOS R&D Analytical Method Validation Platform
Professional Method Validation System for Any Analyte

Features:
- Multi-analyte support
- Session tracking and audit trail
- Persistent state management
- Professional validation workflow
- Export and reporting capabilities

Version: 2.0
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
import hashlib

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="KETOS R&D Method Validation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 4px solid #1e3a8a;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .welcome-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        padding: 2rem 0;
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
    .session-info {
        background-color: #e7f3ff;
        border: 2px solid #1e3a8a;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ValidationSession:
    session_id: str
    method_name: str
    analyte_name: str
    developer_name: str
    start_date: datetime
    last_modified: datetime
    status: str = "in_progress"
    
@dataclass
class ValidationStepConfig:
    step_id: str
    step_number: int
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
    session_id: str
    concentration_range: tuple = None
    num_replicates: int = 3
    num_levels: int = 5
    spike_levels: List[float] = None
    interferents: List[str] = None
    custom_params: Dict = None
    created_at: datetime = None
    
@dataclass
class TestResult:
    session_id: str
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

def generate_session_id(method_name, developer_name, start_date):
    """Generate unique session ID"""
    unique_string = f"{method_name}_{developer_name}_{start_date.isoformat()}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:12]

def initialize_app_state():
    """Initialize base application state"""
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.sessions = {}  # Store all validation sessions
        st.session_state.current_session = None
        st.session_state.show_welcome = True

def initialize_validation_session(session_id):
    """Initialize validation session with all steps"""
    
    # LOC Configuration
    if f'loc_config_{session_id}' not in st.session_state:
        st.session_state[f'loc_config_{session_id}'] = {
            'base_sample_volume': 40.0,
            'extra_volume': 0.0,
            'include_loc_volumes': True,
            'standard_loc': 'LOC15',
            'stock_concentration': 1000.0
        }
    
    # Validation Steps - Universal for any analyte
    if f'validation_steps_{session_id}' not in st.session_state:
        st.session_state[f'validation_steps_{session_id}'] = {
            'linearity': ValidationStepConfig(
                step_id='linearity',
                step_number=1,
                step_name='Step 1: Linearity',
                description='Establish linear relationship between concentration and response',
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
                step_number=2,
                step_name='Step 2: Interference Study',
                description='Test for interference from common ions and substances',
                requires_design=True,
                design_params={
                    'interferent_list': ['Cl⁻', 'SO₄²⁻', 'Ca²⁺', 'Mg²⁺', 'Humic Acid'],
                    'analyte_concentration': 50,
                    'replicates': 3
                },
                expected_tests=60,
                acceptance_criteria={'recovery_range': (90, 110), 'rsd': 5}
            ),
            'repeatability': ValidationStepConfig(
                step_id='repeatability',
                step_number=3,
                step_name='Step 3: Repeatability',
                description='Intra-day precision - same analyst, same conditions',
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
                step_number=4,
                step_name='Step 4: Intermediate Precision',
                description='Inter-day precision - different days, different analysts',
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
                step_number=5,
                step_name='Step 5: Accuracy (Recovery)',
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
                step_number=6,
                step_name='Step 6: LOD & LOQ',
                description='Limit of Detection and Limit of Quantification',
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
                step_number=7,
                step_name='Step 7: Stability',
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
                step_number=8,
                step_name='Step 8: Robustness',
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
                step_number=9,
                step_name='Step 9: Matrix Effects',
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
                step_number=10,
                step_name='Step 10: Working Range',
                description='Establish working range with acceptable precision and accuracy',
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
    
    # Results storage
    if f'designs_{session_id}' not in st.session_state:
        st.session_state[f'designs_{session_id}'] = {}
    
    if f'results_{session_id}' not in st.session_state:
        steps = st.session_state[f'validation_steps_{session_id}']
        st.session_state[f'results_{session_id}'] = {step: [] for step in steps.keys()}
    
    if f'analyses_{session_id}' not in st.session_state:
        st.session_state[f'analyses_{session_id}'] = {}
    
    if f'current_step_{session_id}' not in st.session_state:
        st.session_state[f'current_step_{session_id}'] = 'linearity'

# Initialize app
initialize_app_state()


# ============================================================================
# UTILITY FUNCTIONS (Same as before but session-aware)
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

def save_session_state(session_id):
    """Save session state to browser storage (future: database)"""
    # This will be enhanced with Supabase integration
    session = st.session_state.sessions.get(session_id)
    if session:
        session.last_modified = datetime.now()
        # Future: Save to Supabase
        return True
    return False


# ============================================================================
# WELCOME PAGE & SESSION MANAGEMENT
# ============================================================================

def show_welcome_page():
    """Welcome page for new validation session"""
    
    st.markdown('<div class="welcome-header">🔬 KETOS R&D<br>Analytical Method Validation</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>📋 Professional Method Validation Platform</h3>
    <p>Complete validation workflow for any analytical method and analyte.</p>
    <p><strong>Features:</strong></p>
    <ul>
        <li>10-Step validation protocol</li>
        <li>Session tracking and audit trail</li>
        <li>Multi-analyte support</li>
        <li>Automated statistical analysis</li>
        <li>Professional reporting</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check for existing sessions
    if st.session_state.sessions:
        st.subheader("📂 Existing Validation Sessions")
        
        sessions_data = []
        for sess_id, sess in st.session_state.sessions.items():
            sessions_data.append({
                'Session ID': sess_id,
                'Method Name': sess.method_name,
                'Analyte': sess.analyte_name,
                'Developer': sess.developer_name,
                'Started': sess.start_date.strftime('%Y-%m-%d %H:%M'),
                'Status': sess.status.replace('_', ' ').title()
            })
        
        df_sessions = pd.DataFrame(sessions_data)
        st.dataframe(safe_dataframe_display(df_sessions), width='stretch')
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            session_ids = list(st.session_state.sessions.keys())
            selected_session = st.selectbox(
                "Select Session to Continue",
                [""] + session_ids,
                format_func=lambda x: f"{st.session_state.sessions[x].method_name} - {st.session_state.sessions[x].analyte_name}" if x else "-- Select Session --"
            )
            
            if selected_session and st.button("📂 Open Selected Session", type="primary"):
                st.session_state.current_session = selected_session
                st.session_state.show_welcome = False
                initialize_validation_session(selected_session)
                st.rerun()
        
        with col2:
            if st.button("➕ Start New Validation Session"):
                st.session_state.show_new_session_form = True
                st.rerun()
    
    else:
        st.info("👋 No existing sessions. Start a new validation session below.")
        st.session_state.show_new_session_form = True
    
    # New Session Form
    if st.session_state.get('show_new_session_form', True):
        st.markdown("---")
        st.subheader("➕ Start New Validation Session")
        
        with st.form("new_session_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                method_name = st.text_input(
                    "Method Name *",
                    placeholder="e.g., Spectrophotometric Determination",
                    help="Enter the analytical method name"
                )
                
                analyte_name = st.text_input(
                    "Analyte Name *",
                    placeholder="e.g., Iron (Fe), Nitrate, Phosphate",
                    help="Enter the target analyte"
                )
            
            with col2:
                developer_name = st.text_input(
                    "Developer Name *",
                    placeholder="e.g., John Smith",
                    help="Your full name for audit trail"
                )
                
                start_date = st.date_input(
                    "Validation Start Date *",
                    value=datetime.now(),
                    help="Official start date of validation"
                )
            
            # Optional fields
            st.markdown("**Optional Information:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                concentration_unit = st.text_input("Concentration Unit", value="mg/L")
            
            with col2:
                wavelength = st.text_input("Wavelength (if applicable)", placeholder="e.g., 562 nm")
            
            with col3:
                reference_method = st.text_input("Reference Method", placeholder="e.g., EPA 365.1")
            
            notes = st.text_area(
                "Additional Notes",
                placeholder="Any additional information about this validation...",
                height=100
            )
            
            submitted = st.form_submit_button("🚀 Start Validation Session", type="primary")
            
            if submitted:
                if not method_name or not analyte_name or not developer_name:
                    st.error("❌ Please fill in all required fields (marked with *)")
                else:
                    # Create new session
                    start_datetime = datetime.combine(start_date, datetime.now().time())
                    session_id = generate_session_id(method_name, developer_name, start_datetime)
                    
                    new_session = ValidationSession(
                        session_id=session_id,
                        method_name=method_name,
                        analyte_name=analyte_name,
                        developer_name=developer_name,
                        start_date=start_datetime,
                        last_modified=start_datetime,
                        status="in_progress"
                    )
                    
                    # Store session
                    st.session_state.sessions[session_id] = new_session
                    st.session_state.current_session = session_id
                    
                    # Store optional metadata
                    st.session_state[f'metadata_{session_id}'] = {
                        'concentration_unit': concentration_unit,
                        'wavelength': wavelength,
                        'reference_method': reference_method,
                        'notes': notes
                    }
                    
                    # Initialize validation session
                    initialize_validation_session(session_id)
                    
                    st.session_state.show_welcome = False
                    st.success(f"✅ Validation session created! Session ID: {session_id}")
                    st.balloons()
                    
                    # Save state (future: to database)
                    save_session_state(session_id)
                    
                    st.rerun()

def show_session_info_sidebar():
    """Display current session info in sidebar"""
    if st.session_state.current_session:
        session = st.session_state.sessions[st.session_state.current_session]
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Current Session")
        
        st.sidebar.markdown(f"""
        <div class="session-info">
        <strong>Method:</strong> {session.method_name}<br>
        <strong>Analyte:</strong> {session.analyte_name}<br>
        <strong>Developer:</strong> {session.developer_name}<br>
        <strong>Started:</strong> {session.start_date.strftime('%Y-%m-%d')}<br>
        <strong>Session ID:</strong> <code>{session.session_id}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🏠 Back to Home", use_container_width=True):
            st.session_state.show_welcome = True
            st.rerun()
        
        if st.sidebar.button("💾 Save Session", use_container_width=True):
            if save_session_state(st.session_state.current_session):
                st.sidebar.success("✅ Session saved!")
        
        # Export session button
        if st.sidebar.button("📥 Export Session Data", use_container_width=True):
            export_session_data(st.session_state.current_session)

def export_session_data(session_id):
    """Export all session data to Excel"""
    session = st.session_state.sessions[session_id]
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Session info
        session_info = pd.DataFrame([{
            'Session ID': session.session_id,
            'Method Name': session.method_name,
            'Analyte': session.analyte_name,
            'Developer': session.developer_name,
            'Start Date': session.start_date.strftime('%Y-%m-%d %H:%M'),
            'Last Modified': session.last_modified.strftime('%Y-%m-%d %H:%M'),
            'Status': session.status
        }])
        session_info.to_excel(writer, sheet_name='Session Info', index=False)
        
        # Step status
        steps = st.session_state[f'validation_steps_{session_id}']
        step_status = pd.DataFrame([{
            'Step Number': s.step_number,
            'Step Name': s.step_name,
            'Status': s.status,
            'Expected Tests': s.expected_tests,
            'Collected': len(st.session_state[f'results_{session_id}'].get(s.step_id, []))
        } for s in steps.values()])
        step_status.to_excel(writer, sheet_name='Step Status', index=False)
        
        # Results for each step
        results_dict = st.session_state[f'results_{session_id}']
        for step_id, results in results_dict.items():
            if results:
                df = pd.DataFrame([asdict(r) for r in results])
                step_name = steps[step_id].step_name.replace(':', '-')
                df.to_excel(writer, sheet_name=step_name[:31], index=False)
    
    output.seek(0)
    
    st.sidebar.download_button(
        label="📥 Download Excel Report",
        data=output,
        file_name=f"{session.method_name}_{session.analyte_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ============================================================================
# ANALYSIS FUNCTIONS (Robust, session-aware)
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
        control_data = df[df['metadata'].apply(lambda x: x.get('interferent') == 'None' if isinstance(x, dict) else False)]
        if len(control_data) == 0:
            return {'error': 'No control data found', 'passes': False}
        
        control_mean = control_data['concentration'].mean()
        results_by_interferent = {}
        
        interferents = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('interferent') != 'None')]['metadata'].apply(lambda x: x.get('interferent') if isinstance(x, dict) else None).unique()
        
        for interferent in interferents:
            if interferent:
                int_data = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('interferent') == interferent)]
                
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
    """Analyze intermediate precision"""
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
        t0_data = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('time_point') == 'T0')]
        
        if len(t0_data) == 0:
            return {'error': 'No T0 data found', 'passes': False}
        
        t0_mean = t0_data['concentration'].mean()
        results_by_timepoint = {}
        
        time_points = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('time_point') != 'T0')]['metadata'].apply(lambda x: x.get('time_point') if isinstance(x, dict) else None).unique()
        
        for tp in time_points:
            if tp:
                tp_data = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('time_point') == tp)]
                
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
        normal_data = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('variation') == 'Normal')]
        
        if len(normal_data) == 0:
            return {'error': 'No normal condition data found', 'passes': False}
        
        normal_mean = normal_data['concentration'].mean()
        results_by_parameter = {}
        
        parameters = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('parameter') is not None)]['metadata'].apply(lambda x: x.get('parameter') if isinstance(x, dict) else None).unique()
        
        for param in parameters:
            if param:
                param_results = {}
                
                for var in ['+Δ', '-Δ']:
                    var_data = df[(df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('parameter') == param)) &
                                 (df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('variation') == var))]
                    
                    if len(var_data) > 0:
                        mean = var_data['concentration'].mean()
                        bias = abs((mean - normal_mean) / normal_mean * 100) if normal_mean > 0 else 0
                        
                        param_results[var] = {
                            'mean': float(mean),
                            'bias': float(bias)
                        }
                
                results_by_parameter[param] = param_results
        
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
        di_water = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('matrix') == 'DI Water')]
        
        if len(di_water) == 0:
            return {'error': 'No DI Water control data found', 'passes': False}
        
        di_mean_by_level = {}
        for level in di_water['level_number'].unique():
            di_mean_by_level[int(level)] = di_water[di_water['level_number'] == level]['concentration'].mean()
        
        results_by_matrix = {}
        
        matrices = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('matrix') != 'DI Water')]['metadata'].apply(lambda x: x.get('matrix') if isinstance(x, dict) else None).unique()
        
        for matrix in matrices:
            if matrix:
                matrix_data = df[df['metadata'].apply(lambda x: isinstance(x, dict) and x.get('matrix') == matrix)]
                
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
    """Analyze working range"""
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
# VALIDATION STEP UI FUNCTIONS
# ============================================================================

def show_validation_step(session_id, step_id):
    """Show validation step page - session-aware"""
    
    steps = st.session_state[f'validation_steps_{session_id}']
    step_config = steps[step_id]
    session = st.session_state.sessions[session_id]
    
    st.markdown(f'<div class="step-header">{step_config.step_name}</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    📋 <strong>Purpose:</strong> {step_config.description}<br>
    <strong>Analyte:</strong> {session.analyte_name}
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["📐 Design", "📥 Collect Data", "📊 Analyze", "📈 Results"])
    
    with tabs[0]:
        show_design_tab(session_id, step_id, step_config)
    
    with tabs[1]:
        show_collect_tab(session_id, step_id, step_config)
    
    with tabs[2]:
        show_analyze_tab(session_id, step_id, step_config)
    
    with tabs[3]:
        show_results_tab(session_id, step_id, step_config)

def show_design_tab(session_id, step_id, step_config):
    """Design experiment tab - session-aware"""
    st.subheader("📐 Experiment Design")
    
    session = st.session_state.sessions[session_id]
    designs = st.session_state[f'designs_{session_id}']
    
    if step_id in designs:
        st.markdown("""
        <div class="success-box">
        ✅ <strong>Experiment Already Designed!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        design = designs[step_id]
        
        if design.spike_levels:
            st.write(f"**Number of Levels:** {len(design.spike_levels)}")
            st.write(f"**Replicates per Level:** {design.num_replicates}")
            st.write(f"**Total Tests Required:** {len(design.spike_levels) * design.num_replicates}")
            
            df_levels = pd.DataFrame({
                'Level': range(1, len(design.spike_levels) + 1),
                f'{session.analyte_name} Concentration': [f"{x:.2f}" for x in design.spike_levels]
            })
            st.dataframe(safe_dataframe_display(df_levels), width='stretch')
        
        if design.custom_params:
            st.write("**Additional Parameters:**")
            for key, value in design.custom_params.items():
                st.write(f"- {key}: {value}")
        
        if st.button("🔄 Modify Design"):
            del designs[step_id]
            st.rerun()
    
    else:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Design Required</strong><br>
        Configure your experiment parameters below.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        custom_params = {}
        
        # Get concentration unit from metadata
        metadata = st.session_state.get(f'metadata_{session_id}', {})
        conc_unit = metadata.get('concentration_unit', 'mg/L')
        
        if step_id == 'linearity' or step_id == 'range':
            with col1:
                min_conc = st.number_input(f"Min Concentration ({conc_unit})", 
                                          value=0.0 if step_id == 'linearity' else 5.0, 
                                          step=1.0,
                                          help=f"Lowest {session.analyte_name} concentration")
                max_conc = st.number_input(f"Max Concentration ({conc_unit})", 
                                          value=100.0, 
                                          step=1.0,
                                          help=f"Highest {session.analyte_name} concentration")
            
            with col2:
                num_levels = st.number_input("Number of Levels", 
                                            value=5 if step_id == 'linearity' else 7, 
                                            min_value=3, 
                                            max_value=10)
                num_reps = st.number_input("Replicates per Level", 
                                          value=3, 
                                          min_value=2, 
                                          max_value=10)
            
            spike_levels = np.linspace(min_conc, max_conc, num_levels).tolist()
        
        elif step_id == 'interference':
            with col1:
                analyte_conc = st.number_input(f"{session.analyte_name} Concentration ({conc_unit})", 
                                              value=50.0, 
                                              step=5.0)
                num_reps = st.number_input("Replicates per Condition", 
                                          value=3, 
                                          min_value=2, 
                                          max_value=6)
            
            with col2:
                default_interferents = ['Cl⁻', 'SO₄²⁻', 'Ca²⁺', 'Mg²⁺', 'Humic Acid', 'Turbidity']
                interferents = st.multiselect(
                    "Select Interferents to Test",
                    default_interferents,
                    default=default_interferents[:3],
                    help="Common substances that might interfere with the measurement"
                )
            
            spike_levels = [analyte_conc]
            custom_params = {'interferents': interferents, 'analyte_concentration': analyte_conc}
        
        elif step_id in ['repeatability', 'intermediate', 'accuracy']:
            col1, col2, col3 = st.columns(3)
            with col1:
                low_conc = st.number_input(f"Low Level ({conc_unit})", value=10.0, step=5.0)
            with col2:
                mid_conc = st.number_input(f"Mid Level ({conc_unit})", value=50.0, step=5.0)
            with col3:
                high_conc = st.number_input(f"High Level ({conc_unit})", value=90.0, step=5.0)
            
            spike_levels = [low_conc, mid_conc, high_conc]
            num_levels = 3
            
            if step_id == 'intermediate':
                col1, col2 = st.columns(2)
                with col1:
                    num_days = st.slider("Number of Days", 3, 7, 3)
                with col2:
                    reps_per_day = st.slider("Replicates per Day", 3, 8, 5)
                num_reps = num_days * reps_per_day
                custom_params = {'num_days': num_days, 'reps_per_day': reps_per_day}
            else:
                num_reps = st.slider("Replicates per Level", 
                                    3, 15, 
                                    10 if step_id == 'repeatability' else 5)
        
        elif step_id == 'lod_loq':
            with col1:
                max_low_conc = st.number_input(f"Maximum Low Concentration ({conc_unit})", 
                                              value=10.0, 
                                              step=1.0)
                num_levels = st.slider("Number of Low Levels", 5, 10, 6)
            
            with col2:
                blank_reps = st.slider("Blank Replicates", 7, 15, 10)
                level_reps = st.slider("Replicates per Level", 5, 10, 7)
            
            spike_levels = np.linspace(0, max_low_conc, num_levels).tolist()
            num_reps = level_reps
            custom_params = {'blank_replicates': blank_reps}
        
        elif step_id == 'stability':
            conc = st.number_input(f"Test Concentration ({conc_unit})", value=50.0, step=5.0)
            
            time_points = st.multiselect(
                "Time Points",
                ['T0', '4h', '8h', '24h', '48h', '72h', '1week', '2weeks', '1month'],
                default=['T0', '24h', '48h', '72h', '1week']
            )
            
            storage = st.multiselect(
                "Storage Conditions",
                ['Room Temperature', 'Refrigerated (4°C)', 'Frozen (-20°C)', 'Ambient Light', 'Dark'],
                default=['Room Temperature', 'Refrigerated (4°C)']
            )
            
            num_reps = st.slider("Replicates per Condition", 3, 6, 3)
            
            spike_levels = [conc]
            num_levels = 1
            custom_params = {'time_points': time_points, 'storage_conditions': storage, 'concentration': conc}
        
        elif step_id == 'robustness':
            conc = st.number_input(f"Test Concentration ({conc_unit})", value=50.0, step=5.0)
            
            parameters = st.multiselect(
                "Parameters to Vary",
                ['Temperature (±2°C)', 'pH (±0.5)', 'Reagent Lot', 'Sample Volume (±5%)', 
                 'Mixing Time (±30s)', 'Reaction Time (±2min)', 'Wavelength (±2nm)'],
                default=['Temperature (±2°C)', 'pH (±0.5)', 'Reagent Lot']
            )
            
            num_reps = st.slider("Replicates per Condition", 3, 6, 3)
            
            spike_levels = [conc]
            num_levels = 1
            custom_params = {'parameters': parameters, 'concentration': conc}
        
        elif step_id == 'matrix':
            matrices = st.multiselect(
                "Sample Matrices to Test",
                ['DI Water (Control)', 'Tap Water', 'Surface Water', 'Groundwater', 
                 'Wastewater', 'Seawater', 'Industrial Effluent', 'Drinking Water'],
                default=['DI Water (Control)', 'Tap Water', 'Surface Water', 'Wastewater']
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                low_spike = st.number_input(f"Low Spike ({conc_unit})", value=25.0, step=5.0)
            with col2:
                mid_spike = st.number_input(f"Mid Spike ({conc_unit})", value=50.0, step=5.0)
            with col3:
                high_spike = st.number_input(f"High Spike ({conc_unit})", value=75.0, step=5.0)
            
            spike_levels = [low_spike, mid_spike, high_spike]
            num_levels = 3
            num_reps = st.slider("Replicates per Matrix/Level", 3, 6, 3)
            custom_params = {'matrices': matrices}
        
        else:
            st.error(f"Design interface not implemented for {step_id}")
            return
        
        # Save design button
        st.markdown("---")
        
        total_tests = len(spike_levels) * num_reps
        if custom_params:
            if 'interferents' in custom_params:
                total_tests = len(spike_levels) * num_reps * (len(custom_params['interferents']) + 1)
            elif 'matrices' in custom_params:
                total_tests = len(spike_levels) * num_reps * len(custom_params['matrices'])
            elif 'time_points' in custom_params and 'storage_conditions' in custom_params:
                total_tests = len(custom_params['time_points']) * len(custom_params['storage_conditions']) * num_reps
            elif 'parameters' in custom_params:
                total_tests = len(custom_params['parameters']) * 3 * num_reps  # Normal, +Δ, -Δ
        
        st.info(f"📊 **Total Tests Required:** {total_tests}")
        
        if st.button("💾 Save Experiment Design", type="primary", use_container_width=True):
            design = ExperimentDesign(
                step_id=step_id,
                session_id=session_id,
                concentration_range=(min(spike_levels), max(spike_levels)) if len(spike_levels) > 1 else None,
                num_levels=len(spike_levels),
                num_replicates=num_reps,
                spike_levels=spike_levels,
                custom_params=custom_params if custom_params else None,
                created_at=datetime.now()
            )
            
            designs[step_id] = design
            steps[step_id].status = 'designed'
            
            # Save session
            save_session_state(session_id)
            
            st.success(f"✅ Experiment design saved! Ready to collect {total_tests} tests.")
            st.balloons()
            st.rerun()

# ============================================================================
# COLLECT TAB - Session-aware data collection
# ============================================================================

def show_collect_tab(session_id, step_id, step_config):
    """Data collection tab - session-aware"""
    st.subheader("📥 Data Collection")
    
    session = st.session_state.sessions[session_id]
    designs = st.session_state[f'designs_{session_id}']
    results = st.session_state[f'results_{session_id}']
    loc_config = st.session_state[f'loc_config_{session_id}']
    
    if step_id not in designs:
        st.warning("⚠️ Please design experiment first in the Design tab")
        return
    
    design = designs[step_id]
    step_results = results[step_id]
    
    expected = len(design.spike_levels) * design.num_replicates if design.spike_levels else step_config.expected_tests
    collected = len(step_results)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Tests", expected)
    with col2:
        st.metric("Collected", collected, delta=f"+{collected}" if collected > 0 else None)
    with col3:
        progress_pct = (collected/expected*100) if expected > 0 else 0
        st.metric("Progress", f"{progress_pct:.0f}%")
    
    st.progress(min(collected / expected, 1.0) if expected > 0 else 0)
    
    st.markdown("---")
    
    # File upload
    uploaded_files = st.file_uploader(
        f"Upload JSON files for {session.analyte_name} measurements",
        type=['json'], 
        accept_multiple_files=True,
        help="Upload test result JSON files from your instrument"
    )
    
    if uploaded_files:
        new_results = []
        
        for idx, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name}", expanded=(idx==0)):
                try:
                    content = file.read().decode('utf-8')
                    extracted = extract_json_data(content)
                    
                    if extracted:
                        concentration = calculate_concentration(extracted['loc_doses'], loc_config)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{session.analyte_name} Concentration", 
                                    f"{concentration:.3f}",
                                    help="Calculated from LOC doses")
                        with col2:
                            st.metric("Absorbance", 
                                    f"{extracted['absorbance']:.4f}" if extracted['absorbance'] else "N/A")
                        
                        # Mapping inputs
                        col1, col2 = st.columns(2)
                        with col1:
                            level = st.number_input(
                                "Level Number", 
                                1, len(design.spike_levels), 1, 
                                key=f"lev_{idx}",
                                help="Which concentration level does this test belong to?"
                            )
                        with col2:
                            rep = st.number_input(
                                "Replicate Number", 
                                1, design.num_replicates, 1, 
                                key=f"rep_{idx}",
                                help="Which replicate number is this?"
                            )
                        
                        # Step-specific metadata
                        metadata = {}
                        
                        if step_id == 'interference':
                            interferent = st.selectbox(
                                "Interferent Present",
                                ['None (Control)'] + (design.custom_params.get('interferents', []) if design.custom_params else []),
                                key=f"int_{idx}",
                                help="Select the interferent present in this sample"
                            )
                            metadata = {'interferent': 'None' if interferent == 'None (Control)' else interferent}
                        
                        elif step_id == 'intermediate':
                            test_day = st.selectbox(
                                "Test Day", 
                                list(range(1, design.custom_params.get('num_days', 3) + 1)), 
                                key=f"day_{idx}",
                                help="Which day was this test performed?"
                            )
                            analyst = st.text_input(
                                "Analyst Name",
                                key=f"analyst_{idx}",
                                help="Who performed this test?"
                            )
                            metadata = {'test_day': test_day, 'analyst': analyst}
                        
                        elif step_id == 'stability':
                            time_point = st.selectbox(
                                "Time Point", 
                                design.custom_params.get('time_points', ['T0']), 
                                key=f"time_{idx}",
                                help="When was this sample measured?"
                            )
                            storage = st.selectbox(
                                "Storage Condition", 
                                design.custom_params.get('storage_conditions', ['Room Temperature']), 
                                key=f"stor_{idx}",
                                help="How was this sample stored?"
                            )
                            metadata = {'time_point': time_point, 'storage': storage}
                        
                        elif step_id == 'robustness':
                            parameter = st.selectbox(
                                "Parameter Varied", 
                                ['Normal'] + (design.custom_params.get('parameters', []) if design.custom_params else []), 
                                key=f"param_{idx}",
                                help="Which parameter was varied?"
                            )
                            if parameter != 'Normal':
                                variation = st.selectbox(
                                    "Variation Direction", 
                                    ['Normal', '+Δ', '-Δ'], 
                                    key=f"var_{idx}",
                                    help="Direction of variation"
                                )
                            else:
                                variation = 'Normal'
                            metadata = {'parameter': parameter, 'variation': variation}
                        
                        elif step_id == 'matrix':
                            matrix = st.selectbox(
                                "Sample Matrix", 
                                design.custom_params.get('matrices', ['DI Water (Control)']), 
                                key=f"mat_{idx}",
                                help="What matrix was this sample?"
                            )
                            metadata = {'matrix': matrix.replace(' (Control)', '')}
                        
                        result = TestResult(
                            session_id=session_id,
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
                            replicate_number=rep,
                            metadata=metadata if metadata else None
                        )
                        
                        new_results.append(result)
                        st.success("✅ Ready to import")
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        if new_results:
            st.markdown("---")
            if st.button(f"📥 Import {len(new_results)} Test(s)", type="primary", use_container_width=True):
                results[step_id].extend(new_results)
                steps = st.session_state[f'validation_steps_{session_id}']
                steps[step_id].status = 'collecting'
                
                # Save session
                save_session_state(session_id)
                
                st.success(f"✅ Successfully imported {len(new_results)} tests!")
                st.rerun()
    
    # Display collected data
    if step_results:
        st.markdown("---")
        st.subheader("📊 Collected Data")
        
        metadata_info = st.session_state.get(f'metadata_{session_id}', {})
        conc_unit = metadata_info.get('concentration_unit', 'mg/L')
        
        df = pd.DataFrame([{
            'Test #': r.test_number,
            'Shield Test #': r.shield_test_number,
            'Level': r.level_number,
            'Rep': r.replicate_number,
            f'[{session.analyte_name}] ({conc_unit})': f"{r.concentration:.3f}",
            'Absorbance': f"{r.absorbance:.4f}",
            'Timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M')
        } for r in step_results])
        
        st.dataframe(safe_dataframe_display(df), width='stretch', height=400)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Delete Last Entry", use_container_width=True):
                if step_results:
                    results[step_id].pop()
                    save_session_state(session_id)
                    st.success("✅ Deleted last entry")
                    st.rerun()

# ============================================================================
# ANALYZE TAB - Session-aware analysis
# ============================================================================

def show_analyze_tab(session_id, step_id, step_config):
    """Analysis tab - session-aware"""
    st.subheader("📊 Statistical Analysis")
    
    session = st.session_state.sessions[session_id]
    results = st.session_state[f'results_{session_id}']
    designs = st.session_state[f'designs_{session_id}']
    
    step_results = results[step_id]
    
    if not step_results:
        st.warning("⚠️ No data collected yet. Please collect data in the Collect tab first.")
        return
    
    if step_id not in designs:
        st.error("❌ No experiment design found. Please design the experiment first.")
        return
    
    design = designs[step_id]
    
    # Show data summary
    st.info(f"📊 **Ready to analyze {len(step_results)} test results for {session.analyte_name}**")
    
    # Analysis button
    if st.button("🔬 Run Statistical Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {session.analyte_name} data..."):
            df = pd.DataFrame([asdict(r) for r in step_results])
            
            try:
                # Route to appropriate analysis function
                if step_id == 'linearity':
                    analysis = analyze_linearity(df, design, step_config)
                elif step_id == 'interference':
                    analysis = analyze_interference(df, design, step_config)
                elif step_id == 'repeatability':
                    analysis = analyze_repeatability(df, design, step_config)
                elif step_id == 'intermediate':
                    analysis = analyze_intermediate(df, design, step_config)
                elif step_id == 'accuracy':
                    analysis = analyze_accuracy(df, design, step_config)
                elif step_id == 'lod_loq':
                    analysis = analyze_lod_loq(df, design, step_config)
                elif step_id == 'stability':
                    analysis = analyze_stability(df, design, step_config)
                elif step_id == 'robustness':
                    analysis = analyze_robustness(df, design, step_config)
                elif step_id == 'matrix':
                    analysis = analyze_matrix(df, design, step_config)
                elif step_id == 'range':
                    analysis = analyze_range(df, design, step_config)
                else:
                    analysis = {'error': 'Analysis not implemented for this step'}
                
                if 'error' in analysis:
                    st.error(f"❌ Analysis Error: {analysis['error']}")
                else:
                    # Store analysis results
                    analyses = st.session_state[f'analyses_{session_id}']
                    analyses[step_id] = analysis
                    
                    # Update step status
                    steps = st.session_state[f'validation_steps_{session_id}']
                    steps[step_id].status = 'analyzed'
                    
                    # Save session
                    save_session_state(session_id)
                    
                    st.success("✅ Analysis complete! View results in the Results tab.")
                    st.balloons()
                    st.rerun()
            
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
    
    # Show if already analyzed
    analyses = st.session_state[f'analyses_{session_id}']
    if step_id in analyses:
        st.markdown("---")
        st.success("✅ Analysis completed! View comprehensive results in the **Results** tab →")

# ============================================================================
# RESULTS TAB - Router function
# ============================================================================

def show_results_tab(session_id, step_id, step_config):
    """Results visualization tab - session-aware router"""
    st.subheader("📈 Analysis Results")
    
    session = st.session_state.sessions[session_id]
    analyses = st.session_state[f'analyses_{session_id}']
    
    if step_id not in analyses:
        st.warning("⚠️ No analysis results yet. Please run the analysis in the Analyze tab first.")
        return
    
    analysis = analyses[step_id]
    
    if 'error' in analysis:
        st.error(f"❌ Analysis Error: {analysis['error']}")
        return
    
    # Route to appropriate results display
    if step_id == 'linearity':
        show_linearity_results(session_id, analysis, step_config)
    elif step_id == 'interference':
        show_interference_results(session_id, analysis, step_config)
    elif step_id == 'repeatability':
        show_repeatability_results(session_id, analysis, step_config)
    elif step_id == 'intermediate':
        show_intermediate_results(session_id, analysis, step_config)
    elif step_id == 'accuracy':
        show_accuracy_results(session_id, analysis, step_config)
    elif step_id == 'lod_loq':
        show_lod_loq_results(session_id, analysis, step_config)
    elif step_id == 'stability':
        show_stability_results(session_id, analysis, step_config)
    elif step_id == 'robustness':
        show_robustness_results(session_id, analysis, step_config)
    elif step_id == 'matrix':
        show_matrix_results(session_id, analysis, step_config)
    elif step_id == 'range':
        show_range_results(session_id, analysis, step_config)

# ============================================================================
# RESULTS DISPLAY FUNCTIONS - Session-aware
# All functions receive session_id as first parameter
# ============================================================================

def show_linearity_results(session_id, analysis, config):
    """Display linearity results"""
    session = st.session_state.sessions[session_id]
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    if analysis['passes']:
        st.markdown(f"""
        <div class="success-box">
        ✅ <strong>LINEARITY TEST PASSED</strong><br>
        R² = {analysis['r_squared']:.4f} (Criteria: ≥ {config.acceptance_criteria['r_squared']})
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
        ❌ <strong>LINEARITY TEST FAILED</strong><br>
        R² = {analysis['r_squared']:.4f} (Criteria: ≥ {config.acceptance_criteria['r_squared']})
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² (Coefficient of Determination)", f"{analysis['r_squared']:.4f}")
    with col2:
        st.metric("Slope", f"{analysis['slope']:.6f}")
    with col3:
        st.metric("Intercept", f"{analysis['intercept']:.6f}")
    
    # Calibration curve
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=analysis['x'],
        y=analysis['y'],
        mode='markers',
        name='Measured Points',
        marker=dict(size=12, color='#4A90E2', line=dict(width=2, color='white'))
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis['x'],
        y=analysis['y_pred'],
        mode='lines',
        name='Regression Line',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title=f"Calibration Curve for {session.analyte_name}<br>" + 
              f"<sub>y = {analysis['slope']:.6f}x + {analysis['intercept']:.6f} (R² = {analysis['r_squared']:.4f})</sub>",
        xaxis_title=f"{session.analyte_name} Concentration ({conc_unit})",
        yaxis_title="Absorbance",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=analysis['x'],
            y=analysis['residuals'],
            mode='markers',
            marker=dict(size=10, color='purple')
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_layout(
            title="Residuals Plot",
            xaxis_title=f"Concentration ({conc_unit})",
            yaxis_title="Residuals",
            height=400
        )
        st.plotly_chart(fig_res, use_container_width=True)
    
    with col2:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=analysis['residuals'],
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

def show_interference_results(session_id, analysis, config):
    """Display interference results"""
    session = st.session_state.sessions[session_id]
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    st.info(f"**Control Mean ({session.analyte_name}):** {analysis['control_mean']:.3f} {conc_unit}")
    
    # Summary table
    data = []
    for interferent, res in analysis['by_interferent'].items():
        data.append({
            'Interferent': interferent,
            f'Mean ({conc_unit})': f"{res['mean']:.3f}",
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
        title=f"Recovery of {session.analyte_name} in Presence of Interferents",
        xaxis_title="Interferent",
        yaxis_title="Recovery (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if analysis['passes']:
        st.success(f"✅ No significant interference detected for {session.analyte_name} measurement")
    else:
        st.error("❌ Some interferents show significant interference")

def show_repeatability_results(session_id, analysis, config):
    """Display repeatability results"""
    session = st.session_state.sessions[session_id]
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    if analysis['passes']:
        st.success(f"✅ REPEATABILITY TEST PASSED: All levels meet RSD ≤ {config.acceptance_criteria['rsd']}%")
    else:
        st.error(f"❌ REPEATABILITY TEST FAILED: Some levels exceed RSD {config.acceptance_criteria['rsd']}%")
    
    # Summary table
    data = []
    for level, res in analysis['by_level'].items():
        data.append({
            'Level': level,
            f'Mean ({conc_unit})': f"{res['mean']:.3f}",
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
        title=f"Repeatability of {session.analyte_name} Measurements",
        xaxis_title="Level",
        yaxis_title=f"Concentration ({conc_unit})",
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
        title="RSD by Level",
        xaxis_title="Level",
        yaxis_title="RSD (%)",
        height=400
    )
    
    st.plotly_chart(fig_rsd, use_container_width=True)

def show_intermediate_results(session_id, analysis, config):
    """Display intermediate precision results"""
    session = st.session_state.sessions[session_id]
    st.info(f"**Intermediate Precision:** Assesses precision across multiple days/analysts for {session.analyte_name}")
    show_repeatability_results(session_id, analysis, config)

def show_accuracy_results(session_id, analysis, config):
    """Display accuracy/recovery results"""
    session = st.session_state.sessions[session_id]
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    if analysis['passes']:
        st.success(f"✅ ACCURACY TEST PASSED: All recoveries within {config.acceptance_criteria['recovery_range'][0]}-{config.acceptance_criteria['recovery_range'][1]}%")
    else:
        st.error(f"❌ ACCURACY TEST FAILED: Some recoveries outside acceptable range")
    
    # Summary table
    data = []
    for level, res in analysis['by_level'].items():
        data.append({
            'Level': level,
            f'Spiked ({conc_unit})': f"{res['spike_conc']:.3f}",
            f'Measured ({conc_unit})': f"{res['measured']:.3f}",
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
    
    fig.add_hline(y=100, line_dash="dash", line_color="blue", annotation_text="100%")
    
    fig.update_layout(
        title=f"Recovery of {session.analyte_name} from Spiked Samples",
        xaxis_title="Spike Level",
        yaxis_title="Recovery (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_lod_loq_results(session_id, analysis, config):
    """Display LOD/LOQ results"""
    session = st.session_state.sessions[session_id]
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    st.subheader(f"🔍 Detection and Quantification Limits for {session.analyte_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Blank Mean (Abs)", f"{analysis['blank_mean']:.4f}")
    with col2:
        st.metric("Blank Std Dev", f"{analysis['blank_std']:.4f}")
    with col3:
        st.metric(f"LOD ({conc_unit})", f"{analysis['lod_concentration']:.3f}",
                 help="Limit of Detection = Blank + 3σ")
    with col4:
        st.metric(f"LOQ ({conc_unit})", f"{analysis['loq_concentration']:.3f}",
                 help="Limit of Quantification = Blank + 10σ")
    
    st.markdown("""
    <div class="info-box">
    <strong>Calculation Method:</strong><br>
    • LOD (Limit of Detection) = Blank Mean + 3 × Blank Std Dev<br>
    • LOQ (Limit of Quantification) = Blank Mean + 10 × Blank Std Dev
    </div>
    """, unsafe_allow_html=True)
    
    st.success(f"✅ Detection limits established for {session.analyte_name}")

def show_stability_results(session_id, analysis, config):
    """Display stability results"""
    session = st.session_state.sessions[session_id]
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    st.info(f"**T0 Reference ({session.analyte_name}):** {analysis['t0_mean']:.3f} {conc_unit}")
    
    # Summary table
    data = []
    for tp, res in analysis['by_timepoint'].items():
        data.append({
            'Time Point': tp,
            f'Mean ({conc_unit})': f"{res['mean']:.3f}",
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
        marker=dict(size=12),
        line=dict(width=3, color='#4A90E2')
    ))
    
    upper_limit = analysis['t0_mean'] * (1 + config.acceptance_criteria['deviation'] / 100)
    lower_limit = analysis['t0_mean'] * (1 - config.acceptance_criteria['deviation'] / 100)
    
    fig.add_hrect(
        y0=lower_limit,
        y1=upper_limit,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        annotation_text=f"±{config.acceptance_criteria['deviation']}% Acceptable Range"
    )
    
    fig.update_layout(
        title=f"Stability of {session.analyte_name} Over Time",
        xaxis_title="Time Point",
        yaxis_title=f"Concentration ({conc_unit})",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if analysis['passes']:
        st.success(f"✅ {session.analyte_name} samples stable within acceptable range")
    else:
        st.error(f"❌ {session.analyte_name} shows instability at some time points")

def show_robustness_results(session_id, analysis, config):
    """Display robustness results"""
    session = st.session_state.sessions[session_id]
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    st.info(f"**Normal Condition Mean ({session.analyte_name}):** {analysis['normal_mean']:.3f} {conc_unit}")
    
    for param, variations in analysis['by_parameter'].items():
        with st.expander(f"📊 {param}", expanded=True):
            
            data = []
            for var, res in variations.items():
                data.append({
                    'Variation': var,
                    f'Mean ({conc_unit})': f"{res['mean']:.3f}",
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
                marker_color=['blue'] + ['lightblue'] * len(means),
                text=[f"{analysis['normal_mean']:.3f}"] + [f"{m:.3f}" for m in means],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"{param} - Effect on {session.analyte_name} Measurement",
                xaxis_title="Condition",
                yaxis_title=f"Concentration ({conc_unit})",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    if analysis['passes']:
        st.success(f"✅ Method is robust to parameter variations for {session.analyte_name}")
    else:
        st.error(f"❌ Method shows sensitivity to some parameter variations")

def show_matrix_results(session_id, analysis, config):
    """Display matrix effects results"""
    session = st.session_state.sessions[session_id]
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    st.info(f"**DI Water (Control)** serves as reference for {session.analyte_name} measurement")
    
    for matrix, by_level in analysis['by_matrix'].items():
        with st.expander(f"📊 {matrix}", expanded=True):
            
            data = []
            for level, res in by_level.items():
                data.append({
                    'Level': level,
                    f'DI Water ({conc_unit})': f"{res['di_reference']:.3f}",
                    f'{matrix} ({conc_unit})': f"{res['mean']:.3f}",
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
                name='DI Water (Control)',
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
                title=f"{matrix} vs DI Water for {session.analyte_name}",
                xaxis_title="Level",
                yaxis_title=f"Concentration ({conc_unit})",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    if analysis['passes']:
        st.success(f"✅ No significant matrix effects detected for {session.analyte_name}")
    else:
        st.error(f"❌ Matrix effects detected for {session.analyte_name}")

def show_range_results(session_id, analysis, config):
    """Display working range results"""
    session = st.session_state.sessions[session_id]
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    if analysis['passes']:
        st.success(f"✅ WORKING RANGE VALIDATED: All levels meet acceptance criteria for {session.analyte_name}")
    else:
        st.error(f"❌ WORKING RANGE TEST FAILED: Some levels outside criteria")
    
    # Summary table
    data = []
    for level, res in analysis['by_level'].items():
        data.append({
            'Level': level,
            f'Target ({conc_unit})': f"{res['spike_conc']:.3f}",
            f'Measured ({conc_unit})': f"{res['measured']:.3f}",
            'RSD (%)': f"{res['rsd']:.2f}",
            'Recovery (%)': f"{res['recovery']:.1f}",
            'n': res['n'],
            'Status': '✅' if res['passes'] else '❌'
        })
    
    df = pd.DataFrame(data)
    st.dataframe(safe_dataframe_display(df), width='stretch')
    
    # Recovery across range
    levels = list(analysis['by_level'].keys())
    concentrations = [analysis['by_level'][l]['spike_conc'] for l in levels]
    recoveries = [analysis['by_level'][l]['recovery'] for l in levels]
    colors = ['green' if analysis['by_level'][l]['passes'] else 'red' for l in levels]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"{c:.1f}" for c in concentrations],
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
        title=f"Recovery Across Working Range for {session.analyte_name}",
        xaxis_title=f"Concentration ({conc_unit})",
        yaxis_title="Recovery (%)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Establish working range
    passing_levels = [l for l in levels if analysis['by_level'][l]['passes']]
    if passing_levels:
        min_passing = min([analysis['by_level'][l]['spike_conc'] for l in passing_levels])
        max_passing = max([analysis['by_level'][l]['spike_conc'] for l in passing_levels])
        
        st.markdown(f"""
        <div class="success-box">
        <strong>Validated Working Range for {session.analyte_name}:</strong><br>
        {min_passing:.2f} - {max_passing:.2f} {conc_unit}
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# CONFIG & DASHBOARD PAGES
# ============================================================================

def show_config(session_id):
    """LOC Configuration page - session-aware"""
    st.header("⚙️ LOC Configuration")
    
    session = st.session_state.sessions[session_id]
    loc_config = st.session_state[f'loc_config_{session_id}']
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    conc_unit = metadata.get('concentration_unit', 'mg/L')
    
    st.markdown(f"""
    <div class="info-box">
    📋 <strong>Configure LOC Settings for {session.analyte_name}</strong><br>
    Set up stock concentrations and sample volumes for M₁V₁=M₂V₂ calculations.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Sample Volume")
        
        base_vol = st.number_input(
            "Base Sample Volume (mL)", 
            value=loc_config['base_sample_volume'], 
            step=1.0,
            help="Initial volume of sample before LOC additions"
        )
        extra_vol = st.number_input(
            "Extra Volume (mL)", 
            value=loc_config['extra_volume'], 
            step=0.1,
            help="Additional volume (e.g., reagents)"
        )
        include_loc = st.checkbox(
            "Include LOC volumes in calculation?", 
            value=loc_config['include_loc_volumes'],
            help="Include volumes of LOC additions in total volume"
        )
    
    with col2:
        st.subheader(f"🧪 {session.analyte_name} Standard")
        
        standard_loc = st.selectbox(
            "Standard LOC Channel", 
            [f'LOC{i}' for i in range(1, 17)], 
            index=[f'LOC{i}' for i in range(1, 17)].index(loc_config['standard_loc']),
            help="Which LOC contains the analyte standard?"
        )
        stock_conc = st.number_input(
            f"Stock Concentration ({conc_unit})", 
            value=loc_config['stock_concentration'], 
            step=10.0,
            help=f"Concentration of {session.analyte_name} in the stock solution"
        )
    
    # Example calculation
    st.markdown("---")
    st.subheader("📊 Example Calculation")
    
    example_loc = st.number_input("Example LOC dose (μL)", value=50.0, step=10.0)
    
    total_vol = base_vol + extra_vol
    if include_loc:
        total_vol += (example_loc / 1000)
    
    example_conc = stock_conc * (example_loc / 1000) / total_vol
    
    st.success(f"""
    **{session.analyte_name} Concentration Calculation:**
    - Stock: {stock_conc} {conc_unit}
    - LOC Dose: {example_loc} μL = {example_loc/1000} mL
    - Total Volume: {total_vol:.3f} mL
    - **Final [{session.analyte_name}]: {example_conc:.3f} {conc_unit}**
    """)
    
    st.markdown("---")
    
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        loc_config.update({
            'base_sample_volume': base_vol,
            'extra_volume': extra_vol,
            'include_loc_volumes': include_loc,
            'standard_loc': standard_loc,
            'stock_concentration': stock_conc
        })
        
        save_session_state(session_id)
        st.success("✅ Configuration saved!")
        st.balloons()

def show_dashboard(session_id):
    """Overview dashboard - session-aware"""
    st.header("📊 Validation Dashboard")
    
    session = st.session_state.sessions[session_id]
    steps = st.session_state[f'validation_steps_{session_id}']
    results = st.session_state[f'results_{session_id}']
    analyses = st.session_state[f'analyses_{session_id}']
    
    # Session summary
    st.markdown(f"""
    <div class="session-info">
    <h3>📋 Validation Session Summary</h3>
    <strong>Method:</strong> {session.method_name}<br>
    <strong>Analyte:</strong> {session.analyte_name}<br>
    <strong>Developer:</strong> {session.developer_name}<br>
    <strong>Started:</strong> {session.start_date.strftime('%Y-%m-%d %H:%M')}<br>
    <strong>Last Modified:</strong> {session.last_modified.strftime('%Y-%m-%d %H:%M')}<br>
    <strong>Session ID:</strong> <code>{session.session_id}</code>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Progress metrics
    completed = sum(1 for s in steps.values() if s.status == 'analyzed')
    total = len(steps)
    in_progress = sum(1 for s in steps.values() if s.status in ['designed', 'collecting'])
    total_tests = sum(len(r) for r in results.values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Steps", total)
    with col2:
        st.metric("Completed", completed, delta=f"{(completed/total*100):.0f}%")
    with col3:
        st.metric("In Progress", in_progress)
    with col4:
        st.metric("Total Tests Collected", total_tests)
    
    st.progress(completed / total if total > 0 else 0)
    
    st.markdown("---")
    
    # Steps status table
    st.subheader("📋 Validation Steps Status")
    
    status_data = []
    for step_id, step_config in steps.items():
        results_count = len(results.get(step_id, []))
        analyzed = '✅' if step_config.status == 'analyzed' else '⚪'
        
        # Pass/Fail status
        pass_status = ''
        if step_id in analyses:
            analysis = analyses[step_id]
            if 'error' not in analysis:
                pass_status = '✅ PASS' if analysis.get('passes', False) else '❌ FAIL'
        
        status_data.append({
            'Step': step_config.step_name,
            'Status': step_config.status.replace('_', ' ').title(),
            'Tests Collected': results_count,
            'Expected': step_config.expected_tests,
            'Progress': f"{(results_count/step_config.expected_tests*100):.0f}%" if step_config.expected_tests > 0 else "N/A",
            'Analyzed': analyzed,
            'Result': pass_status
        })
    
    df_status = pd.DataFrame(status_data)
    st.dataframe(safe_dataframe_display(df_status), width='stretch', height=400)
    
    # Progress visualization
    st.markdown("---")
    st.subheader("📈 Progress by Step")
    
    steps_list = [s['Step'] for s in status_data]
    collected = [s['Tests Collected'] for s in status_data]
    expected = [s['Expected'] for s in status_data]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Collected',
        x=steps_list,
        y=collected,
        marker_color='#28a745',
        text=collected,
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Remaining',
        x=steps_list,
        y=[e - c for e, c in zip(expected, collected)],
        marker_color='#ffc107',
        text=[e - c for e, c in zip(expected, collected)],
        textposition='inside'
    ))
    
    fig.update_layout(
        barmode='stack',
        xaxis_tickangle=-45,
        height=500,
        yaxis_title="Number of Tests",
        title=f"Test Collection Progress for {session.analyte_name}"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pass/Fail Summary
    if analyses:
        st.markdown("---")
        st.subheader("✅ Analysis Results Summary")
        
        passed = sum(1 for a in analyses.values() if 'error' not in a and a.get('passes', False))
        failed = sum(1 for a in analyses.values() if 'error' not in a and not a.get('passes', False))
        errors = sum(1 for a in analyses.values() if 'error' in a)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Passed", passed, delta="✅")
        with col2:
            st.metric("Failed", failed, delta="❌" if failed > 0 else None)
        with col3:
            st.metric("Errors", errors, delta="⚠️" if errors > 0 else None)
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export & Reporting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Complete Validation Report", type="primary", use_container_width=True):
            export_complete_report(session_id)
    
    with col2:
        if st.button("💾 Export Raw Data", use_container_width=True):
            export_raw_data(session_id)

def export_complete_report(session_id):
    """Export complete validation report with all data and analyses"""
    session = st.session_state.sessions[session_id]
    steps = st.session_state[f'validation_steps_{session_id}']
    results = st.session_state[f'results_{session_id}']
    analyses = st.session_state[f'analyses_{session_id}']
    metadata = st.session_state.get(f'metadata_{session_id}', {})
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Session info
        session_info = pd.DataFrame([{
            'Session ID': session.session_id,
            'Method Name': session.method_name,
            'Analyte': session.analyte_name,
            'Developer': session.developer_name,
            'Start Date': session.start_date.strftime('%Y-%m-%d %H:%M'),
            'Last Modified': session.last_modified.strftime('%Y-%m-%d %H:%M'),
            'Status': session.status,
            'Concentration Unit': metadata.get('concentration_unit', 'mg/L'),
            'Wavelength': metadata.get('wavelength', 'N/A'),
            'Reference Method': metadata.get('reference_method', 'N/A')
        }])
        session_info.to_excel(writer, sheet_name='Session Info', index=False)
        
        # Step status summary
        step_status = pd.DataFrame([{
            'Step Number': s.step_number,
            'Step Name': s.step_name,
            'Status': s.status,
            'Expected Tests': s.expected_tests,
            'Collected': len(results.get(s.step_id, [])),
            'Analyzed': 'Yes' if s.step_id in analyses else 'No',
            'Pass/Fail': ('PASS' if analyses[s.step_id].get('passes', False) else 'FAIL') 
                        if s.step_id in analyses and 'error' not in analyses[s.step_id] else 'N/A'
        } for s in steps.values()])
        step_status.to_excel(writer, sheet_name='Validation Summary', index=False)
        
        # Results for each step
        for step_id, step_results in results.items():
            if step_results:
                df = pd.DataFrame([asdict(r) for r in step_results])
                step_name = steps[step_id].step_name.replace(':', '-').replace('/', '-')
                df.to_excel(writer, sheet_name=f"{step_name[:25]}_Data", index=False)
        
        # Analysis results for each step
        for step_id, analysis in analyses.items():
            if 'error' not in analysis:
                step_name = steps[step_id].step_name.replace(':', '-').replace('/', '-')
                
                # Create summary based on analysis type
                if step_id == 'linearity':
                    df_analysis = pd.DataFrame([{
                        'Metric': 'R²',
                        'Value': analysis['r_squared'],
                        'Criteria': analysis['criteria']['r_squared'],
                        'Pass': analysis['passes']
                    }, {
                        'Metric': 'Slope',
                        'Value': analysis['slope'],
                        'Criteria': 'N/A',
                        'Pass': 'N/A'
                    }, {
                        'Metric': 'Intercept',
                        'Value': analysis['intercept'],
                        'Criteria': 'N/A',
                        'Pass': 'N/A'
                    }])
                else:
                    df_analysis = pd.DataFrame([{
                        'Analysis': step_name,
                        'Overall Result': 'PASS' if analysis['passes'] else 'FAIL',
                        'Details': str(analysis.get('criteria', {}))
                    }])
                
                df_analysis.to_excel(writer, sheet_name=f"{step_name[:25]}_Analysis", index=False)
    
    output.seek(0)
    
    filename = f"{session.method_name}_{session.analyte_name}_Validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filename = filename.replace(' ', '_').replace('/', '-')
    
    st.download_button(
        label="📥 Download Complete Validation Report",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    
    st.success(f"✅ Report ready for download: {filename}")

def export_raw_data(session_id):
    """Export raw data only"""
    session = st.session_state.sessions[session_id]
    steps = st.session_state[f'validation_steps_{session_id}']
    results = st.session_state[f'results_{session_id}']
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for step_id, step_results in results.items():
            if step_results:
                df = pd.DataFrame([asdict(r) for r in step_results])
                step_name = steps[step_id].step_name.replace(':', '-').replace('/', '-')
                df.to_excel(writer, sheet_name=step_name[:31], index=False)
    
    output.seek(0)
    
    filename = f"{session.analyte_name}_RawData_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filename = filename.replace(' ', '_')
    
    st.download_button(
        label="📥 Download Raw Data",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
