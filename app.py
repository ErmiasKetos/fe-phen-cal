"""
Fe²⁺/Fe³⁺ Method Validation Dashboard
Professional Streamlit Application

Features:
- JSON file upload and processing
- LOC configuration management
- Real-time validation progress tracking
- Interactive data visualization
- M1V1=M2V2 concentration calculations
- Statistical analysis
- Export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fe Method Validation Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'loc_config' not in st.session_state:
    st.session_state.loc_config = {
        'base_sample_volume': 40.0,
        'extra_volume': 0.0,
        'include_loc_volumes': True,
        'standard_loc': 'LOC15',
        'stock_concentration': 1000.0,
        'loc_reagents': {}
    }

if 'imported_data' not in st.session_state:
    st.session_state.imported_data = []

if 'validation_progress' not in st.session_state:
    st.session_state.validation_progress = {
        '4A_Linearity_Phase1': {'total': 36, 'completed': 0},
        '4A_Linearity_Phase2': {'total': 40, 'completed': 0},
        '4B_Interference': {'total': 108, 'completed': 0},
        '5A_Repeatability': {'total': 30, 'completed': 0},
        '5A_Intermediate_Precision': {'total': 45, 'completed': 0},
        '5B_Accuracy': {'total': 36, 'completed': 0},
        '5C_LOD_LOQ': {'total': 10, 'completed': 0},
        '5D_Stability': {'total': 48, 'completed': 0},
        '6_Robustness': {'total': 24, 'completed': 0},
        '7_Sample_Matrix': {'total': 36, 'completed': 0}
    }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_json_data(json_content):
    """Extract Shield Test #, LOC doses, and absorbance from JSON"""
    try:
        data = json.loads(json_content) if isinstance(json_content, str) else json_content
        
        # Extract Shield Test Number
        shield_test_num = None
        if 'payload' in data:
            shield_test_num = (data['payload'].get('exp_number') or 
                             data['payload'].get('test_number') or 
                             data['payload'].get('testNumber'))
        
        if not shield_test_num:
            shield_test_num = (data.get('exp_number') or 
                             data.get('test_number') or 
                             data.get('testNumber'))
        
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
            scan_type = ''
            if 'parameters' in scan:
                scan_type = scan['parameters'].get('scanType', '').lower()
            else:
                scan_type = scan.get('scanType', '').lower()
            
            if 'back' in scan_type or 'bg' in scan_type:
                bg_scan = scan
            elif 'sample' in scan_type:
                sample_scan = scan  # Keep updating to get LAST sample
        
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
        
        # Extract LOC doses from last sample scan
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
    
    # Filter outliers (3 * 1.4826 * MAD threshold)
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
    
    # Get spike volume for standard LOC
    spike_volume_uL = loc_doses.get(standard_loc, 0)
    
    if spike_volume_uL == 0:
        return 0.0
    
    # Calculate total volume
    total_volume_mL = base_volume + extra_volume
    
    if include_loc:
        total_loc_volume_uL = sum(loc_doses.values())
        total_volume_mL += (total_loc_volume_uL / 1000)
    
    # M1V1 = M2V2
    spike_volume_mL = spike_volume_uL / 1000
    concentration = stock_conc * (spike_volume_mL / total_volume_mL)
    
    return concentration

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🧪 Fe²⁺/Fe³⁺ Method Validation Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["📊 Dashboard", "⚙️ LOC Configuration", "📥 Data Import", 
         "📈 Validation Progress", "📉 Statistics & Analysis", "📋 Export Data"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Quick Stats:**
    - Total Tests: 413
    - Completed: {completed}
    - Progress: {progress:.1f}%
    """.format(
        completed=sum(v['completed'] for v in st.session_state.validation_progress.values()),
        progress=sum(v['completed'] for v in st.session_state.validation_progress.values()) / 413 * 100
    ))
    
    # Route to pages
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "⚙️ LOC Configuration":
        show_loc_configuration()
    elif page == "📥 Data Import":
        show_data_import()
    elif page == "📈 Validation Progress":
        show_validation_progress()
    elif page == "📉 Statistics & Analysis":
        show_statistics()
    elif page == "📋 Export Data":
        show_export()

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================

def show_dashboard():
    st.header("📊 Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_tests = 413
    completed_tests = sum(v['completed'] for v in st.session_state.validation_progress.values())
    pending_tests = total_tests - completed_tests
    progress_pct = (completed_tests / total_tests * 100) if total_tests > 0 else 0
    
    with col1:
        st.metric("Total Tests", total_tests, help="Total validation test specifications")
    
    with col2:
        st.metric("Completed", completed_tests, 
                 delta=f"{progress_pct:.1f}%", delta_color="normal")
    
    with col3:
        st.metric("Pending", pending_tests)
    
    with col4:
        st.metric("Files Imported", len(st.session_state.imported_data))
    
    st.markdown("---")
    
    # Progress by validation step
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Validation Progress by Step")
        
        # Create progress DataFrame
        progress_data = []
        for step, data in st.session_state.validation_progress.items():
            progress_data.append({
                'Validation Step': step.replace('_', ' '),
                'Completed': data['completed'],
                'Total': data['total'],
                'Progress': (data['completed'] / data['total'] * 100) if data['total'] > 0 else 0
            })
        
        df_progress = pd.DataFrame(progress_data)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Completed',
            x=df_progress['Validation Step'],
            y=df_progress['Completed'],
            marker_color='#28a745'
        ))
        fig.add_trace(go.Bar(
            name='Pending',
            x=df_progress['Validation Step'],
            y=df_progress['Total'] - df_progress['Completed'],
            marker_color='#ffc107'
        ))
        
        fig.update_layout(
            barmode='stack',
            xaxis_tickangle=-45,
            height=400,
            showlegend=True,
            title="Tests by Validation Step"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Overall Progress")
        
        # Pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Completed', 'Pending'],
            values=[completed_tests, pending_tests],
            hole=.5,
            marker_colors=['#28a745', '#ffc107']
        )])
        
        fig.update_layout(
            height=400,
            annotations=[dict(text=f'{progress_pct:.1f}%', x=0.5, y=0.5, 
                            font_size=30, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent imports
    if st.session_state.imported_data:
        st.markdown("---")
        st.subheader("📥 Recent Imports")
        
        df_recent = pd.DataFrame(st.session_state.imported_data[-10:])
        df_recent = df_recent[['timestamp', 'shield_test_number', 'concentration', 
                               'absorbance', 'temperature']]
        df_recent.columns = ['Timestamp', 'Shield Test #', 'Concentration (mg/L)', 
                            'Absorbance', 'Temp (°C)']
        
        st.dataframe(df_recent, use_container_width=True)

# ============================================================================
# PAGE: LOC CONFIGURATION
# ============================================================================

def show_loc_configuration():
    st.header("⚙️ LOC Configuration")
    
    st.markdown("""
    <div class="info-box">
    📋 <strong>Configure your LOC settings for M₁V₁=M₂V₂ calculations</strong><br>
    Set the stock concentration for your Fe standard and specify which LOC position contains it.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Sample Volume Configuration")
        
        base_volume = st.number_input(
            "Base Sample Volume (mL)",
            min_value=0.0,
            max_value=1000.0,
            value=st.session_state.loc_config['base_sample_volume'],
            step=1.0,
            help="Initial water sample volume in measurement cell"
        )
        
        extra_volume = st.number_input(
            "Extra Constant Volume (mL)",
            min_value=0.0,
            max_value=100.0,
            value=st.session_state.loc_config['extra_volume'],
            step=0.1,
            help="Fixed volume from valves/manifold (usually 0)"
        )
        
        include_loc = st.checkbox(
            "Include LOC dosing volumes in total?",
            value=st.session_state.loc_config['include_loc_volumes'],
            help="Add all LOC dosing volumes to the total sample volume"
        )
    
    with col2:
        st.subheader("🧪 Standard LOC Configuration")
        
        standard_loc = st.selectbox(
            "Which LOC contains the Fe standard?",
            options=[f'LOC{i}' for i in range(1, 17)],
            index=14,  # LOC15 default
            help="Select the LOC position that contains your Fe standard solution"
        )
        
        stock_conc = st.number_input(
            "Stock Concentration (mg/L)",
            min_value=0.0,
            max_value=10000.0,
            value=st.session_state.loc_config['stock_concentration'],
            step=10.0,
            help="Concentration of your Fe standard stock solution"
        )
    
    # Update session state
    if st.button("💾 Save Configuration", type="primary"):
        st.session_state.loc_config.update({
            'base_sample_volume': base_volume,
            'extra_volume': extra_volume,
            'include_loc_volumes': include_loc,
            'standard_loc': standard_loc,
            'stock_concentration': stock_conc
        })
        
        st.markdown("""
        <div class="success-box">
        ✅ <strong>Configuration saved successfully!</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Show calculation example
    st.markdown("---")
    st.subheader("🧮 Calculation Preview")
    
    st.markdown(f"""
    **Current Configuration:**
    - Stock Concentration: **{stock_conc} mg/L** ({standard_loc})
    - Base Sample Volume: **{base_volume} mL**
    - Extra Volume: **{extra_volume} mL**
    - Include LOC Volumes: **{'Yes' if include_loc else 'No'}**
    
    **Example Calculation:**
    
    If {standard_loc} doses **100 µL** and other LOCs dose **900 µL total**:
    
    ```
    V_spike = 100 µL = 0.1 mL
    V_total = {base_volume} mL + {extra_volume} mL{' + 1.0 mL (LOCs)' if include_loc else ''} = {base_volume + extra_volume + (1.0 if include_loc else 0)} mL
    
    C_final = {stock_conc} × (0.1 / {base_volume + extra_volume + (1.0 if include_loc else 0)})
            = {stock_conc * (0.1 / (base_volume + extra_volume + (1.0 if include_loc else 0))):.3f} mg/L
    ```
    """)

# ============================================================================
# PAGE: DATA IMPORT
# ============================================================================

def show_data_import():
    st.header("📥 Data Import")
    
    st.markdown("""
    <div class="info-box">
    📁 <strong>Upload JSON files from your Fe validation experiments</strong><br>
    Files will be processed automatically to extract Shield Test #, LOC dosing, and absorbance data.
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose JSON file(s)",
        type=['json'],
        accept_multiple_files=True,
        help="Select one or more JSON files to import"
    )
    
    if uploaded_files:
        st.subheader(f"📂 Processing {len(uploaded_files)} file(s)...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing: {file.name}")
            
            try:
                # Read and parse JSON
                content = file.read().decode('utf-8')
                extracted = extract_json_data(content)
                
                if extracted:
                    # Calculate concentration
                    concentration = calculate_concentration(
                        extracted['loc_doses'],
                        st.session_state.loc_config
                    )
                    
                    extracted['concentration'] = concentration
                    extracted['filename'] = file.name
                    
                    # Format LOC doses string
                    loc_str = ', '.join([f"{k}:{v}" for k, v in extracted['loc_doses'].items()])
                    extracted['loc_doses_str'] = loc_str if loc_str else 'None'
                    
                    # Add to session state
                    st.session_state.imported_data.append(extracted)
                    
                    # Update progress (for demo, just increment first step)
                    first_step = list(st.session_state.validation_progress.keys())[0]
                    if st.session_state.validation_progress[first_step]['completed'] < \
                       st.session_state.validation_progress[first_step]['total']:
                        st.session_state.validation_progress[first_step]['completed'] += 1
                    
                    results.append({
                        'File': file.name,
                        'Status': '✅ Success',
                        'Shield Test #': extracted['shield_test_number'],
                        'Concentration': f"{concentration:.3f} mg/L",
                        'LOC Doses': extracted['loc_doses_str']
                    })
                else:
                    results.append({
                        'File': file.name,
                        'Status': '❌ Failed',
                        'Shield Test #': 'N/A',
                        'Concentration': 'N/A',
                        'LOC Doses': 'N/A'
                    })
            
            except Exception as e:
                results.append({
                    'File': file.name,
                    'Status': f'❌ Error: {str(e)}',
                    'Shield Test #': 'N/A',
                    'Concentration': 'N/A',
                    'LOC Doses': 'N/A'
                })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ Processing complete!")
        
        # Show results
        st.markdown("---")
        st.subheader("📊 Import Results")
        
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
        
        success_count = df_results['Status'].str.contains('Success').sum()
        fail_count = len(df_results) - success_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Successfully imported: {success_count} files")
        with col2:
            if fail_count > 0:
                st.error(f"❌ Failed: {fail_count} files")

# (Continuing in next message due to length...)

# ============================================================================
# PAGE: VALIDATION PROGRESS
# ============================================================================

def show_validation_progress():
    st.header("📈 Validation Progress Details")
    
    # Progress table
    progress_data = []
    for step, data in st.session_state.validation_progress.items():
        progress_pct = (data['completed'] / data['total'] * 100) if data['total'] > 0 else 0
        progress_data.append({
            'Validation Step': step.replace('_', ' '),
            'Total Tests': data['total'],
            'Completed': data['completed'],
            'Pending': data['total'] - data['completed'],
            'Progress (%)': f"{progress_pct:.1f}%",
            'Status': '✅ Complete' if data['completed'] >= data['total'] else 
                     '🔄 In Progress' if data['completed'] > 0 else '⚪ Not Started'
        })
    
    df_progress = pd.DataFrame(progress_data)
    
    st.dataframe(
        df_progress.style.background_gradient(subset=['Completed'], cmap='Greens'),
        use_container_width=True,
        hide_index=True
    )
    
    # Timeline visualization
    st.markdown("---")
    st.subheader("🕐 Progress Timeline")
    
    fig = go.Figure()
    
    for idx, row in df_progress.iterrows():
        completed = row['Completed']
        total = row['Total Tests']
        
        fig.add_trace(go.Bar(
            name=row['Validation Step'],
            x=[completed],
            y=[row['Validation Step']],
            orientation='h',
            marker=dict(color='#28a745'),
            text=f"{completed}/{total}",
            textposition='inside',
            hovertemplate=f"<b>{row['Validation Step']}</b><br>" +
                         f"Completed: {completed}/{total}<br>" +
                         f"Progress: {row['Progress (%)']}<extra></extra>"
        ))
    
    fig.update_layout(
        barmode='relative',
        height=500,
        showlegend=False,
        xaxis_title="Tests Completed",
        yaxis_title="",
        xaxis=dict(range=[0, df_progress['Total Tests'].max() * 1.1])
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: STATISTICS & ANALYSIS
# ============================================================================

def show_statistics():
    st.header("📉 Statistics & Analysis")
    
    if not st.session_state.imported_data:
        st.warning("⚠️ No data imported yet. Please import JSON files first.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.imported_data)
    
    # Summary statistics
    st.subheader("📊 Concentration Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{df['concentration'].mean():.3f} mg/L")
    
    with col2:
        st.metric("Std Dev", f"{df['concentration'].std():.3f} mg/L")
    
    with col3:
        st.metric("Min", f"{df['concentration'].min():.3f} mg/L")
    
    with col4:
        st.metric("Max", f"{df['concentration'].max():.3f} mg/L")
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Concentration Distribution")
        fig = px.histogram(
            df,
            x='concentration',
            nbins=20,
            title="Concentration Frequency",
            labels={'concentration': 'Concentration (mg/L)', 'count': 'Frequency'}
        )
        fig.update_traces(marker_color='#4A90E2')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Absorbance Distribution")
        fig = px.histogram(
            df,
            x='absorbance',
            nbins=20,
            title="Absorbance Frequency",
            labels={'absorbance': 'Absorbance', 'count': 'Frequency'}
        )
        fig.update_traces(marker_color='#764ba2')
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    st.markdown("---")
    st.subheader("📅 Data Over Time")
    
    df_sorted = df.sort_values('timestamp')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_sorted['timestamp'],
        y=df_sorted['concentration'],
        mode='lines+markers',
        name='Concentration',
        line=dict(color='#4A90E2', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Concentration vs Time",
        xaxis_title="Timestamp",
        yaxis_title="Concentration (mg/L)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("---")
    st.subheader("🔗 Correlation Analysis")
    
    numeric_cols = ['concentration', 'absorbance', 'bg_mean', 'sample_mean', 'temperature']
    df_numeric = df[numeric_cols].dropna()
    
    if len(df_numeric) > 0:
        corr_matrix = df_numeric.corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title="Correlation Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed data table
    st.markdown("---")
    st.subheader("📋 Detailed Data Table")
    
    df_display = df[['timestamp', 'shield_test_number', 'concentration', 
                     'absorbance', 'temperature', 'loc_doses_str']].copy()
    df_display.columns = ['Timestamp', 'Shield Test #', 'Conc (mg/L)', 
                         'Absorbance', 'Temp (°C)', 'LOC Doses']
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: EXPORT DATA
# ============================================================================

def show_export():
    st.header("📋 Export Data")
    
    if not st.session_state.imported_data:
        st.warning("⚠️ No data to export yet. Please import JSON files first.")
        return
    
    st.markdown("""
    <div class="info-box">
    💾 <strong>Export your validation data</strong><br>
    Download your data in various formats for reporting and analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare DataFrame
    df = pd.DataFrame(st.session_state.imported_data)
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Excel Export")
        
        # Create Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            df_summary = df[['shield_test_number', 'concentration', 'absorbance', 
                           'bg_mean', 'sample_mean', 'temperature', 'loc_doses_str']]
            df_summary.columns = ['Shield Test #', 'Concentration (mg/L)', 'Absorbance',
                                 'BG Mean', 'Sample Mean', 'Temperature (°C)', 'LOC Doses']
            df_summary.to_excel(writer, sheet_name='Validation Data', index=False)
            
            # Progress sheet
            progress_data = []
            for step, data in st.session_state.validation_progress.items():
                progress_data.append({
                    'Validation Step': step,
                    'Total Tests': data['total'],
                    'Completed': data['completed'],
                    'Pending': data['total'] - data['completed']
                })
            pd.DataFrame(progress_data).to_excel(writer, sheet_name='Progress', index=False)
        
        output.seek(0)
        
        st.download_button(
            label="📥 Download Excel",
            data=output,
            file_name=f"fe_validation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        st.subheader("📄 CSV Export")
        
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"fe_validation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        st.subheader("🔢 JSON Export")
        
        json_str = json.dumps(st.session_state.imported_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"fe_validation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Preview
    st.markdown("---")
    st.subheader("👀 Data Preview")
    
    st.dataframe(df.head(20), use_container_width=True)
    
    st.info(f"📊 Total records: {len(df)}")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
