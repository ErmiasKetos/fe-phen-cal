
import streamlit as st
import pandas as pd
import numpy as np
import json, io, zipfile, datetime as dt
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Fe²⁺/Fe³⁺ Method Validation",
    page_icon="🧪",
    layout="wide",
)

# -----------------------------
# Utility
# -----------------------------

def robust_mean(values):
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad == 0:
        return float(np.mean(vals))
    thr = 3 * 1.4826 * mad
    keep = np.abs(vals - med) <= thr
    kept = vals[keep]
    if kept.size == 0:
        kept = vals
    return float(np.mean(kept))

def parse_values(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for v in raw:
            if isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, str):
                try:
                    out.append(float(v))
                except:
                    pass
            elif isinstance(v, dict):
                for k in ("value","intensity"):
                    if k in v:
                        try:
                            out.append(float(v[k]))
                        except:
                            pass
        return [x for x in out if np.isfinite(x)]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except:
                pass
        return [x for x in out if np.isfinite(x)]
    return []

def find_scans(obj):
    # recursively find 'scans'
    if isinstance(obj, dict):
        if "scans" in obj and isinstance(obj["scans"], list):
            return obj["scans"]
        for v in obj.values():
            res = find_scans(v)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for it in obj:
            res = find_scans(it)
            if res is not None:
                return res
    return None

def get_scan_type(scan):
    stype = (
        scan.get("scanType")
        or (scan.get("parameters") or {}).get("scanType")
        or (scan.get("parameters") or {}).get("scan_type")
        or ""
    )
    stype = str(stype).lower()
    if "back" in stype or "bg" in stype:
        return "background"
    if "sam" in stype:
        return "sample"
    return ""

def get_channel_array(scan, channel_key):
    if not isinstance(scan, dict):
        return []
    # direct hit
    if channel_key in scan:
        return parse_values(scan[channel_key])
    # case-insensitive
    for k in scan.keys():
        if str(k).lower() == str(channel_key).lower():
            return parse_values(scan[k])
    # nested common
    nested = scan.get("channels") or scan.get("Channels") or scan.get("data") or scan.get("Data")
    if isinstance(nested, dict):
        if channel_key in nested:
            return parse_values(nested[channel_key])
        for k in nested.keys():
            if str(k).lower() == str(channel_key).lower():
                return parse_values(nested[k])
    return []

def extract_loc_doses(scan):
    doses = {}
    if not isinstance(scan, dict):
        return doses
    for src in (scan, scan.get("parameters", {})):
        if isinstance(src, dict):
            for k, v in src.items():
                if isinstance(k, str) and k.upper().startswith("LOC"):
                    try:
                        val = float(v)
                        if val > 0:
                            doses[k.upper()] = val
                    except:
                        pass
    return doses

def compute_absorbance(bg_vals, smp_vals):
    bgm = robust_mean(bg_vals)
    sm = robust_mean(smp_vals)
    if bgm is None or sm is None or sm <= 0:
        return None, bgm, sm
    A = np.log10(bgm / sm)
    return float(A), float(bgm), float(sm)

# -----------------------------
# Session state initialization
# -----------------------------

DEFAULT_SHEETS = [
    "4A_Linearity_Phase1",
    "4A_Linearity_Phase2",
    "4B_Interference",
    "5A_Repeatability",
    "5A_Intermediate_Precision",
    "5B_Accuracy",
    "5C_LOD_LOQ",
    "5D_Stability",
    "6_Robustness",
    "7_Sample_Matrix",
]

if "loc_config" not in st.session_state:
    # LOC1..LOC16 table
    rows = []
    for i in range(1,17):
        rows.append({
            "LOC": f"LOC{i}",
            "Reagent Role": "Not Used" if i!=15 else "Fe Standard",
            "Stock Conc (mg/L)": 1000.0 if i==15 else "",
            "Is Standard?": True if i==15 else False,
            "Description": "Fe²⁺ stock 1000 mg/L" if i==15 else "",
            "Notes": "",
        })
    st.session_state.loc_config = pd.DataFrame(rows)
if "vol_config" not in st.session_state:
    st.session_state.vol_config = {
        "Base Sample Volume (mL)": 40.0,
        "Extra Constant Volume (mL)": 0.0,
        "Include LOC Volumes?": True,
        "LED Channel": "SC_Green",
    }
if "raw_log" not in st.session_state:
    st.session_state.raw_log = pd.DataFrame(columns=[
        "Import Time","Shield Test #","File Name","Calculated Conc (mg/L)","LOC Doses (µL)",
        "Absorbance","BG Mean","Sample Mean","Temperature (°C)","LED Channel","Import Status","Target Sheet","Row Inserted","Notes"
    ])
if "sheets" not in st.session_state:
    st.session_state.sheets = {name: pd.DataFrame() for name in DEFAULT_SHEETS}

# Pre-seed linearity sheets with template rows
def ensure_linearity_templates():
    # L1: 12 concentrations × 3 reps
    name = "4A_Linearity_Phase1"
    if st.session_state.sheets[name].empty:
        concentrations = [0,1,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25]
        data = []
        tid = 1
        for c in concentrations:
            for r in range(1,4):
                data.append({
                    "Test #": f"L1-{tid:03d}",
                    "Shield Test #": "",
                    "Date": "",
                    "Target Conc (mg/L)": c,
                    "LOC Dosing (µL)": "",
                    "Replicate": r,
                    "Absorbance": "",
                    "BG Mean": "",
                    "Sample Mean": "",
                    "Temp (°C)": "",
                    "Status": "PENDING",
                    "Notes": ""
                })
                tid+=1
        st.session_state.sheets[name] = pd.DataFrame(data)
    # L2: 8 concentrations × 5 reps
    name2 = "4A_Linearity_Phase2"
    if st.session_state.sheets[name2].empty:
        concentrations = [0,2.5,5,10,15,20,22.5,25]
        data = []
        tid = 1
        for c in concentrations:
            for r in range(1,6):
                data.append({
                    "Test #": f"L2-{tid:03d}",
                    "Shield Test #": "",
                    "Date": "",
                    "Target Conc (mg/L)": c,
                    "LOC Dosing (µL)": "",
                    "Replicate": r,
                    "Absorbance": "",
                    "BG Mean": "",
                    "Sample Mean": "",
                    "Temp (°C)": "",
                    "Status": "PENDING",
                    "Notes": ""
                })
                tid+=1
        st.session_state.sheets[name2] = pd.DataFrame(data)

ensure_linearity_templates()

# -----------------------------
# Header
# -----------------------------
st.title("🧪 Fe²⁺/Fe³⁺ Method Validation — Streamlit")
st.caption("Complete validation workspace: LOC configuration, JSON import, auto-concentration, linked validation sheets, and dashboard.")

tabs = st.tabs(["Setup","Import & Process JSON","Validation Sheets","Control Panel","Raw Import Log","Project"])

# -----------------------------
# Setup
# -----------------------------
with tabs[0]:
    st.subheader("⚙️ LOC Configuration & Volumes")
    col1, col2 = st.columns((2,1))
    with col1:
        st.markdown("**LOC mapping (mark exactly one standard and set its stock concentration)**")
        edited = st.data_editor(
            st.session_state.loc_config,
            column_config={
                "Is Standard?": st.column_config.CheckboxColumn("Is Standard?"),
                "Stock Conc (mg/L)": st.column_config.NumberColumn(format="%.6f"),
            },
            use_container_width=True,
            num_rows="fixed",
            height=420,
        )
        st.session_state.loc_config = edited
    with col2:
        st.markdown("**Volumes & channel**")
        vc = st.session_state.vol_config
        vc["Base Sample Volume (mL)"] = st.number_input("Base Sample Volume (mL)", 1.0, 200.0, float(vc["Base Sample Volume (mL)"]), 0.1)
        vc["Extra Constant Volume (mL)"] = st.number_input("Extra Constant Volume (mL)", 0.0, 50.0, float(vc["Extra Constant Volume (mL)"]), 0.1)
        vc["Include LOC Volumes?"] = st.toggle("Include LOC Volumes?", value=bool(vc["Include LOC Volumes?"]))
        vc["LED Channel"] = st.text_input("LED Channel key", value=vc["LED Channel"])
        st.session_state.vol_config = vc

    st.divider()
    st.markdown("**Concentration calculation preview (M₁V₁ = M₂V₂)**")
    # example preview using first standard LOC with 100 µL spike & sum LOC 0.8 mL
    std_df = st.session_state.loc_config[st.session_state.loc_config["Is Standard?"]==True]
    if len(std_df)==1 and pd.notna(std_df.iloc[0]["Stock Conc (mg/L)"]):
        cstock = float(std_df.iloc[0]["Stock Conc (mg/L)"])
        Vsp = 0.1 # mL
        sum_loc = 0.8 if st.session_state.vol_config["Include LOC Volumes?"] else 0.0
        Vtot = st.session_state.vol_config["Base Sample Volume (mL)"] + st.session_state.vol_config["Extra Constant Volume (mL)"] + sum_loc
        cfinal = cstock * (Vsp / Vtot)
        st.info(f"Example: Stock={cstock:g} mg/L, Spike=0.1 mL, Total={Vtot:g} mL → **{cfinal:.3f} mg/L**")
    else:
        st.warning("Mark one standard LOC and enter its stock concentration to preview.")

# -----------------------------
# Import & Process
# -----------------------------
def compute_conc_from_loc(loc_doses):
    # Find standard
    std_rows = st.session_state.loc_config[st.session_state.loc_config["Is Standard?"]==True]
    if len(std_rows)!=1 or pd.isna(std_rows.iloc[0]["Stock Conc (mg/L)"]):
        return 0.0
    std_loc = std_rows.iloc[0]["LOC"]
    stock = float(std_rows.iloc[0]["Stock Conc (mg/L)"])
    spike_uL = float(loc_doses.get(std_loc, 0.0))
    if spike_uL<=0:
        return 0.0
    vc = st.session_state.vol_config
    Vtot = vc["Base Sample Volume (mL)"] + vc["Extra Constant Volume (mL)"]
    if vc["Include LOC Volumes?"]:
        Vtot += sum(loc_doses.values())/1000.0
    Vsp = spike_uL/1000.0
    return float(stock * (Vsp / Vtot))

def parse_device_json(file_bytes, fname):
    try:
        data = json.loads(file_bytes.decode("utf-8"))
    except Exception as e:
        return {"error": f"Invalid JSON: {e}"}

    scans = find_scans(data) or []
    bg = None; smp = None
    for sc in scans:
        stype = get_scan_type(sc)
        if stype=="background" and bg is None:
            bg = sc
        elif stype=="sample":
            smp = sc  # keep last sample

    channel = st.session_state.vol_config["LED Channel"]
    bg_vals = get_channel_array(bg or {}, channel)
    smp_vals = get_channel_array(smp or {}, channel)
    A, bgm, sm = compute_absorbance(bg_vals, smp_vals)

    doses = extract_loc_doses(smp or {})
    conc = compute_conc_from_loc(doses)

    # temperature
    temp = None
    if isinstance(smp, dict) and "bb_temp" in smp:
        try: temp=float(smp["bb_temp"])
        except: pass
    elif isinstance(data, dict):
        payload = data.get("payload", {})
        try: temp=float(payload.get("bb_temp", None))
        except: pass

    # shield test number
    stn = ""
    payload = data.get("payload", {})
    for key in ("exp_number","test_number","testNumber","shield_test_number"):
        if key in payload:
            stn = str(payload[key]); break
    if not stn:
        for key in ("exp_number","test_number","testNumber","shield_test_number"):
            if key in data:
                stn = str(data[key]); break

    return {
        "shield": stn,
        "absorbance": A,
        "bg_mean": bgm,
        "sample_mean": sm,
        "temp": temp,
        "loc_doses": doses,
        "calc_conc": conc,
        "channel": channel,
    }

def append_raw_log(info, filename, status="INSERTED", target_sheet="", row_inserted=""):
    log = st.session_state.raw_log
    newrow = {
        "Import Time": dt.datetime.now().isoformat(timespec="seconds"),
        "Shield Test #": info.get("shield",""),
        "File Name": filename,
        "Calculated Conc (mg/L)": info.get("calc_conc",0.0),
        "LOC Doses (µL)": ", ".join(f"{k}:{int(v)}" for k,v in info.get("loc_doses",{}).items()) or "None",
        "Absorbance": info.get("absorbance", None),
        "BG Mean": info.get("bg_mean", None),
        "Sample Mean": info.get("sample_mean", None),
        "Temperature (°C)": info.get("temp", None),
        "LED Channel": info.get("channel",""),
        "Import Status": status,
        "Target Sheet": target_sheet,
        "Row Inserted": row_inserted,
        "Notes": "",
    }
    st.session_state.raw_log = pd.concat([log, pd.DataFrame([newrow])], ignore_index=True)

def auto_insert(info):
    conc = info.get("calc_conc", 0.0)
    target = "4A_Linearity_Phase1" if conc <= 25 else "7_Sample_Matrix"
    df = st.session_state.sheets[target]
    if df.empty:
        return target, ""
    # find row where Target Conc matches and Absorbance empty
    idx = None
    if "Target Conc (mg/L)" in df.columns:
        with np.errstate(invalid='ignore'):
            tc = pd.to_numeric(df["Target Conc (mg/L)"], errors="coerce")
        candidates = df.index[tc == float(np.round(conc,6))].tolist()
        for i in candidates:
            if pd.isna(df.loc[i,"Absorbance"]) or df.loc[i,"Absorbance"]=="":
                idx = i; break
    if idx is None:
        # append at end with minimal columns
        newrow = {c:"" for c in df.columns}
        if "Target Conc (mg/L)" in newrow:
            newrow["Target Conc (mg/L)"]=conc
        df = pd.concat([df, pd.DataFrame([newrow])], ignore_index=True)
        idx = len(df)-1
    # write values
    if "Shield Test #"] in df.columns:
        df.loc[idx,"Shield Test #"] = info.get("shield","")
    if "Date" in df.columns:
        df.loc[idx,"Date"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    if "LOC Dosing (µL)" in df.columns:
        df.loc[idx,"LOC Dosing (µL)"] = ", ".join(f"{k}:{int(v)}" for k,v in info.get("loc_doses",{}).items()) or ""
    if "Absorbance" in df.columns:
        df.loc[idx,"Absorbance"] = info.get("absorbance","")
    if "BG Mean" in df.columns:
        df.loc[idx,"BG Mean"] = info.get("bg_mean","")
    if "Sample Mean" in df.columns:
        df.loc[idx,"Sample Mean"] = info.get("sample_mean","")
    if "Temp (°C)" in df.columns:
        df.loc[idx,"Temp (°C)"] = info.get("temp","")
    if "Status" in df.columns:
        df.loc[idx,"Status"] = "COMPLETE"
    st.session_state.sheets[target] = df
    return target, str(idx+2)  # approximated row

with tabs[1]:
    st.subheader("📥 Import & Process JSON")
    up = st.file_uploader("Upload one or more device JSON files", type=["json"], accept_multiple_files=True)
    if up:
        for f in up:
            info = parse_device_json(f.read(), f.name)
            if "error" in info:
                st.error(f"{f.name}: {info['error']}")
                append_raw_log({"channel": st.session_state.vol_config["LED Channel"]}, f.name, status="FAILED")
                continue
            tgt, row = auto_insert(info)
            append_raw_log(info, f.name, status="INSERTED", target_sheet=tgt, row_inserted=row)
        st.success("Import complete. See Raw Import Log and Validation Sheets.")

# -----------------------------
# Validation Sheets
# -----------------------------
with tabs[2]:
    st.subheader("📊 Validation Sheets")
    choose = st.selectbox("Select sheet", DEFAULT_SHEETS, index=0)
    df = st.session_state.sheets.get(choose, pd.DataFrame())
    edited = st.data_editor(df, use_container_width=True, height=420, num_rows="dynamic")
    st.session_state.sheets[choose] = edited
    colA, colB = st.columns(2)
    with colA:
        st.download_button("Download sheet as CSV", data=edited.to_csv(index=False), file_name=f"{choose}.csv", mime="text/csv")
    with colB:
        # simple trend plot if columns exist
        if "Target Conc (mg/L)" in edited.columns and "Absorbance" in edited.columns:
            try:
                x = pd.to_numeric(edited["Target Conc (mg/L)"], errors="coerce")
                y = pd.to_numeric(edited["Absorbance"], errors="coerce")
                mask = x.notna() & y.notna()
                fig, ax = plt.subplots(figsize=(5,3.2))
                ax.scatter(x[mask], y[mask])
                ax.set_xlabel("Conc (mg/L)"); ax.set_ylabel("Absorbance (A)"); ax.set_title("Calibration points (sheet view)")
                st.pyplot(fig)
            except Exception as e:
                st.caption(f"Plot unavailable: {e}")

# -----------------------------
# Control Panel
# -----------------------------
def progress_summary():
    rows = []
    mapping = {
        "4A_Linearity_Phase1":36,
        "4A_Linearity_Phase2":40,
        "4B_Interference":108,
        "5A_Repeatability":30,
        "5A_Intermediate_Precision":45,
        "5B_Accuracy":36,
        "5C_LOD_LOQ":10,
        "5D_Stability":48,
        "6_Robustness":24,
        "7_Sample_Matrix":36,
    }
    for name, total in mapping.items():
        df = st.session_state.sheets.get(name, pd.DataFrame())
        complete = 0
        if not df.empty and "Status" in df.columns:
            complete = int((df["Status"]=="COMPLETE").sum())
        rows.append([name, total, complete, total-complete, (complete/total if total>0 else 0.0)])
    res = pd.DataFrame(rows, columns=["Validation Step","Total Tests","Completed","Pending","Progress %"])
    return res

with tabs[3]:
    st.subheader("🎯 Control Panel & Progress")
    cp = progress_summary()
    st.dataframe(cp, use_container_width=True)
    st.metric("Files Imported", len(st.session_state.raw_log))
    st.metric("Overall Progress", f"{100*cp['Progress %'].mean():.1f}%")

# -----------------------------
# Raw Import Log
# -----------------------------
with tabs[4]:
    st.subheader("📋 Raw Import Log")
    st.dataframe(st.session_state.raw_log, use_container_width=True, height=420)
    st.download_button("Download log CSV", data=st.session_state.raw_log.to_csv(index=False), file_name="raw_import_log.csv", mime="text/csv")

# -----------------------------
# Project Save/Load
# -----------------------------
def export_project_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        # config
        z.writestr("loc_config.csv", st.session_state.loc_config.to_csv(index=False))
        z.writestr("vol_config.json", json.dumps(st.session_state.vol_config, indent=2))
        # sheets
        for name, df in st.session_state.sheets.items():
            z.writestr(f"sheets/{name}.csv", df.to_csv(index=False))
        # log
        z.writestr("raw_import_log.csv", st.session_state.raw_log.to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()

def import_project_zip(uploaded):
    try:
        zf = zipfile.ZipFile(io.BytesIO(uploaded.read()))
        # configs
        if "loc_config.csv" in zf.namelist():
            st.session_state.loc_config = pd.read_csv(zf.open("loc_config.csv"))
        if "vol_config.json" in zf.namelist():
            st.session_state.vol_config = json.loads(zf.read("vol_config.json"))
        # sheets
        for name in DEFAULT_SHEETS:
            p = f"sheets/{name}.csv"
            if p in zf.namelist():
                st.session_state.sheets[name] = pd.read_csv(zf.open(p))
        # log
        if "raw_import_log.csv" in zf.namelist():
            st.session_state.raw_log = pd.read_csv(zf.open("raw_import_log.csv"))
        st.success("Project restored.")
    except Exception as e:
        st.error(f"Failed to import project: {e}")

with tabs[5]:
    st.subheader("📦 Project Export / Import")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Project ZIP", data=export_project_zip(), file_name="fe_validation_project.zip", mime="application/zip")
    with col2:
        upz = st.file_uploader("Import Project ZIP", type=["zip"], key="proj_zip")
        if upz:
            import_project_zip(upz)

st.markdown("---")
st.caption("Tip: This app expects device JSON files that contain both Background and Sample scans and an LED channel (default: SC_Green). It uses the last Sample scan for dosing and absorbance calculations.")
