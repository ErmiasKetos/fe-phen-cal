
import streamlit as st
import pandas as pd
import numpy as np
import json, io, zipfile, datetime as dt
import matplotlib.pyplot as plt
from scipy import stats

# =============================
# APP CONFIG
# =============================
st.set_page_config(page_title="R&D MD and Analysis", page_icon="🧪", layout="wide")

SECTIONS = [
    "1) Linearity",
    "2) Interference Study",
    "3) Repeatability",
    "4) Intermediate Precision",
    "5) Accuracy / Recovery",
    "6) LOD / LOQ",
    "7) Stability",
    "8) Robustness",
    "9) Sample Matrix Effects",
    "Project & Settings"
]

DEFAULT_LED = "SC_Green"

# =============================
# UTILITIES
# =============================
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
    kept = vals[keep] if keep.any() else vals
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
    if channel_key in scan:
        return parse_values(scan[channel_key])
    for k in scan.keys():
        if str(k).lower() == str(channel_key).lower():
            return parse_values(scan[k])
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

def compute_conc_from_loc(loc_doses, loc_config, vol_config):
    std_rows = loc_config[loc_config["Is Standard?"]==True]
    if len(std_rows)!=1 or pd.isna(std_rows.iloc[0]["Stock Conc (mg/L)"]):
        return 0.0
    std_loc = std_rows.iloc[0]["LOC"]
    stock = float(std_rows.iloc[0]["Stock Conc (mg/L)"])
    spike_uL = float(loc_doses.get(std_loc, 0.0))
    if spike_uL<=0:
        return 0.0
    Vtot = vol_config["Base Sample Volume (mL)"] + vol_config["Extra Constant Volume (mL)"]
    if vol_config["Include LOC Volumes?"]:
        Vtot += sum(loc_doses.values())/1000.0
    Vsp = spike_uL/1000.0
    return float(stock * (Vsp / Vtot))

def save_fig_download(fig, base):
    png = io.BytesIO()
    pdf = io.BytesIO()
    fig.savefig(png, format="png", bbox_inches="tight", dpi=160)
    png.seek(0)
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    pdf.seek(0)
    st.download_button("Download PNG", png, file_name=f"{base}.png", mime="image/png")
    st.download_button("Download PDF", pdf, file_name=f"{base}.pdf", mime="application/pdf")

# =============================
# SESSION STATE
# =============================
if "loc_config" not in st.session_state:
    rows = []
    for i in range(1,17):
        rows.append({
            "LOC": f"LOC{i}",
            "Reagent Role": "Not Used" if i!=15 else "Standard",
            "Stock Conc (mg/L)": 1000.0 if i==15 else "",
            "Is Standard?": True if i==15 else False,
            "Description": "Primary stock (e.g., Fe 1000 mg/L)" if i==15 else "",
            "Notes": "",
        })
    st.session_state.loc_config = pd.DataFrame(rows)

if "vol_config" not in st.session_state:
    st.session_state.vol_config = {
        "Base Sample Volume (mL)": 40.0,
        "Extra Constant Volume (mL)": 0.0,
        "Include LOC Volumes?": True,
        "LED Channel": DEFAULT_LED,
    }

def ensure_store(key):
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame()

# One store per validation step
for key in [
    "store_linearity", "store_interference", "store_repeatability",
    "store_intermediate", "store_accuracy", "store_lodloq",
    "store_stability", "store_robustness", "store_matrix"
]:
    ensure_store(key)

if "raw_log" not in st.session_state:
    st.session_state.raw_log = pd.DataFrame(columns=[
        "Import Time","Shield Test #","File Name","Calculated Conc (mg/L)","LOC Doses (µL)",
        "Absorbance","BG Mean","Sample Mean","Temperature (°C)","LED Channel","Import Status","Target Section","Row Inserted","Notes"
    ])

# =============================
# DOE GENERATORS
# =============================
def doe_linearity(levels, replicates):
    rows = []
    tid = 1
    for c in levels:
        for r in range(1, replicates+1):
            rows.append({
                "Run": f"L-{tid:03d}",
                "Target Conc (mg/L)": c,
                "Replicate": r,
                "Absorbance": "",
                "BG Mean": "",
                "Sample Mean": "",
                "Temperature (°C)": "",
                "Status": "PENDING",
                "Notes": ""
            })
            tid += 1
    return pd.DataFrame(rows)

def doe_interference(interferents, interf_levels, fe_levels, replicates):
    rows = []
    tid = 1
    for interferent in interferents:
        for il in interf_levels:
            for fe in fe_levels:
                for r in range(1, replicates+1):
                    rows.append({
                        "Run": f"INT-{tid:03d}",
                        "Interferent": interferent,
                        "Interf. Level": il,
                        "Fe Level (mg/L)": fe,
                        "Replicate": r,
                        "Absorbance": "",
                        "Expected Abs": "",
                        "Recovery %": "",
                        "Status": "PENDING",
                        "Notes": ""
                    })
                    tid += 1
    return pd.DataFrame(rows)

def doe_repeatability(levels, replicates):
    rows = []
    tid = 1
    for c in levels:
        for r in range(1, replicates+1):
            rows.append({
                "Run": f"REP-{tid:03d}",
                "Conc (mg/L)": c,
                "Replicate": r,
                "Absorbance": "",
                "Status": "PENDING"
            })
            tid += 1
    return pd.DataFrame(rows)

def doe_intermediate(levels, days, reps_per_day):
    rows = []
    tid = 1
    for d in range(1, days+1):
        for c in levels:
            for r in range(1, reps_per_day+1):
                rows.append({
                    "Run": f"IP-{tid:03d}",
                    "Day": d,
                    "Conc (mg/L)": c,
                    "Replicate": r,
                    "Absorbance": "",
                    "Status": "PENDING"
                })
                tid += 1
    return pd.DataFrame(rows)

def doe_accuracy(matrices, spike_levels, replicates):
    rows = []
    tid = 1
    for m in matrices:
        for s in spike_levels:
            for r in range(1, replicates+1):
                rows.append({
                    "Run": f"ACC-{tid:03d}",
                    "Matrix": m,
                    "Spike (mg/L)": s,
                    "Replicate": r,
                    "Measured (mg/L)": "",
                    "Expected (mg/L)": s,
                    "Recovery %": "",
                    "Status": "PENDING"
                })
                tid += 1
    return pd.DataFrame(rows)

def doe_lodloq(blank_reps=10, low_levels=(0.5,1.0,1.5), reps=3):
    rows = []
    # blanks
    for i in range(1, blank_reps+1):
        rows.append({"Run": f"BLK-{i:02d}", "Type": "Blank", "Conc (mg/L)": 0.0, "Replicate": 1, "Absorbance": ""})
    # low concs
    tid = 1
    for lv in low_levels:
        for r in range(1, reps+1):
            rows.append({"Run": f"LOW-{tid:02d}", "Type": "Low", "Conc (mg/L)": lv, "Replicate": r, "Absorbance": ""})
            tid += 1
    return pd.DataFrame(rows)

def doe_stability(levels, timepoints, replicates=2):
    rows = []
    tid = 1
    for c in levels:
        for t in timepoints:
            for r in range(1,replicates+1):
                rows.append({
                    "Run": f"STAB-{tid:03d}",
                    "Conc (mg/L)": c,
                    "Time (min)": t,
                    "Replicate": r,
                    "Absorbance": "",
                    "% Change": "",
                    "Status": "PENDING"
                })
                tid += 1
    return pd.DataFrame(rows)

def doe_robustness(factors, test_conc=12.5, tests_per_factor=3):
    rows = []
    tid = 1
    for f_name, nominal, testval in factors:
        for i in range(tests_per_factor):
            rows.append({
                "Run": f"ROB-{tid:03d}",
                "Factor": f_name,
                "Nominal": nominal,
                "Test Setting": testval,
                "Conc (mg/L)": test_conc,
                "Absorbance": "",
                "% Difference": "",
                "Status": "PENDING"
            })
            tid += 1
    return pd.DataFrame(rows)

def doe_matrix(matrices, levels, replicates=3):
    rows = []
    tid = 1
    for m in matrices:
        for c in levels:
            for r in range(1,replicates+1):
                rows.append({
                    "Run": f"MAT-{tid:03d}",
                    "Matrix": m,
                    "Conc (mg/L)": c,
                    "Replicate": r,
                    "Absorbance": "",
                    "DI Absorbance": "",
                    "Matrix Effect %": "",
                    "Status": "PENDING"
                })
                tid += 1
    return pd.DataFrame(rows)

# =============================
# JSON PARSING
# =============================
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
            smp = sc  # last Sample

    channel = st.session_state.vol_config["LED Channel"]
    bg_vals = get_channel_array(bg or {}, channel)
    smp_vals = get_channel_array(smp or {}, channel)
    A, bgm, sm = compute_absorbance(bg_vals, smp_vals)

    doses = extract_loc_doses(smp or {})
    conc = compute_conc_from_loc(doses, st.session_state.loc_config, st.session_state.vol_config)

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
        "file_name": fname
    }

def append_raw_log(info, status="INSERTED", target_section="", row_inserted=""):
    log = st.session_state.raw_log
    newrow = {
        "Import Time": dt.datetime.now().isoformat(timespec="seconds"),
        "Shield Test #": info.get("shield",""),
        "File Name": info.get("file_name",""),
        "Calculated Conc (mg/L)": info.get("calc_conc",0.0),
        "LOC Doses (µL)": ", ".join(f"{k}:{int(v)}" for k,v in info.get("loc_doses",{}).items()) or "None",
        "Absorbance": info.get("absorbance", None),
        "BG Mean": info.get("bg_mean", None),
        "Sample Mean": info.get("sample_mean", None),
        "Temperature (°C)": info.get("temp", None),
        "LED Channel": info.get("channel",""),
        "Import Status": status,
        "Target Section": target_section,
        "Row Inserted": row_inserted,
        "Notes": "",
    }
    st.session_state.raw_log = pd.concat([log, pd.DataFrame([newrow])], ignore_index=True)

# =============================
# ANALYSIS ROUTINES
# =============================
def fit_linear(conc, abs_, weights=None):
    x = np.asarray(conc, dtype=float)
    y = np.asarray(abs_, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2:
        return None
    X = np.vstack([x, np.ones_like(x)]).T
    if weights is not None:
        w = np.asarray(weights, dtype=float)[m]
        W = np.diag(w)
        beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    slope, intercept = beta[0], beta[1]
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    return {"slope": slope, "intercept": intercept, "R2": R2, "x": x, "y": y, "yhat": yhat, "ss_res": ss_res}

def precision_stats(df, conc_col="Conc (mg/L)", abs_col="Absorbance", group_cols=None):
    res = []
    if group_cols is None:
        group_cols = [conc_col]
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = pd.to_numeric(sub[abs_col], errors="coerce").dropna()
        if len(y)>0:
            mean = y.mean(); sd = y.std(ddof=1) if len(y)>1 else 0.0
            rsd = (sd/mean*100) if mean!=0 else np.nan
            res.append(list(keys) + [len(y), mean, sd, rsd])
    cols = list(group_cols) + ["n","Mean","SD","%RSD"]
    return pd.DataFrame(res, columns=cols)

def accuracy_stats(df, meas_col="Measured (mg/L)", exp_col="Expected (mg/L)", group_cols=None):
    res = []
    if group_cols is None:
        group_cols = [exp_col]
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        m = pd.to_numeric(sub[meas_col], errors="coerce")
        e = pd.to_numeric(sub[exp_col], errors="coerce")
        rec = (m/e*100).replace([np.inf,-np.inf], np.nan)
        if rec.notna().any():
            res.append(list(keys) + [len(rec.dropna()), rec.mean(), rec.std(ddof=1) if rec.count()>1 else 0.0])
    cols = list(group_cols) + ["n","Recovery % (mean)","SD"]
    return pd.DataFrame(res, columns=cols)

def lod_loq_from_blanks(blank_abs, slope):
    b = np.asarray(blank_abs, dtype=float)
    b = b[np.isfinite(b)]
    if b.size == 0 or not np.isfinite(slope) or slope==0:
        return np.nan, np.nan, np.nan
    sd_b = np.std(b, ddof=1) if b.size>1 else np.std(b)
    lod = (3*sd_b)/slope
    loq = (10*sd_b)/slope
    return sd_b, lod, loq

# =============================
# HEADER / NAV
# =============================
st.title("R&D MD and Analysis")
st.caption("Workflow-based validation for colorimetric/spectrophotometric methods with device JSON ingestion.")

with st.sidebar:
    st.header("Navigation")
    section = st.radio("Go to", SECTIONS, index=0)
    st.markdown("---")
    st.subheader("Global Settings")
    with st.expander("LOC Configuration"):
        st.dataframe(st.session_state.loc_config, use_container_width=True, height=250)
    with st.expander("Volumes & Channel"):
        vc = st.session_state.vol_config
        vc["Base Sample Volume (mL)"] = st.number_input("Base Sample Volume (mL)", 1.0, 200.0, float(vc["Base Sample Volume (mL)"]), 0.1, key="base_vol")
        vc["Extra Constant Volume (mL)"] = st.number_input("Extra Constant Volume (mL)", 0.0, 50.0, float(vc["Extra Constant Volume (mL)"]), 0.1, key="ext_vol")
        vc["Include LOC Volumes?"] = st.toggle("Include LOC Volumes?", value=bool(vc["Include LOC Volumes?"]), key="incl_loc")
        vc["LED Channel"] = st.text_input("LED Channel key", value=vc["LED Channel"], key="ledkey")
        st.session_state.vol_config = vc

# =============================
# SECTION 1: LINEARITY
# =============================
if section.startswith("1"):
    st.subheader("1) Linearity – DOE → Data → Analysis")

    with st.expander("Design of Experiments (DOE)", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            rng = st.selectbox("Auto-generate levels", ["Custom","0–25 mg/L (12 pts)","0–25 mg/L (8 pts)"], index=0)
            if rng=="0–25 mg/L (12 pts)":
                levels = [0,1,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25]
            elif rng=="0–25 mg/L (8 pts)":
                levels = [0,2.5,5,10,15,20,22.5,25]
            else:
                txt = st.text_input("Manual levels (mg/L, comma-separated)", "0,1,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25")
                levels = [float(x) for x in txt.split(",")]
        with colB:
            reps = st.number_input("Replicates per level", 1, 20, 3)

        if st.button("Generate DOE", key="gen_lin"):
            st.session_state.store_linearity = doe_linearity(levels, reps)

    st.dataframe(st.session_state.store_linearity, use_container_width=True, height=350)

    st.markdown("**Import device JSON to populate Absorbance/means automatically**")
    up = st.file_uploader("Upload JSON (Background + Sample in one file)", type=["json"], accept_multiple_files=True, key="lin_json")
    if up:
        for f in up:
            info = parse_device_json(f.read(), f.name)
            if "error" in info:
                st.error(f"{f.name}: {info['error']}")
                append_raw_log(info, status="FAILED", target_section="Linearity")
                continue
            # find first matching row (by Target Conc) without Absorbance
            df = st.session_state.store_linearity
            conc = np.round(info.get("calc_conc", 0.0), 6)
            idx = None
            if "Target Conc (mg/L)" in df.columns:
                tc = pd.to_numeric(df["Target Conc (mg/L)"], errors="coerce")
                for i in df.index[tc == conc]:
                    if df.loc[i, "Absorbance"] in ("", np.nan):
                        idx = i; break
            if idx is None and not df.empty:
                idx = df.index[-1]  # fallback
            if idx is not None:
                df.loc[idx, "Absorbance"] = info.get("absorbance", "")
                df.loc[idx, "BG Mean"] = info.get("bg_mean","")
                df.loc[idx, "Sample Mean"] = info.get("sample_mean","")
                df.loc[idx, "Temperature (°C)"] = info.get("temp","")
                df.loc[idx, "Status"] = "COMPLETE"
                st.session_state.store_linearity = df
                append_raw_log(info, status="INSERTED", target_section="Linearity", row_inserted=str(idx+2))

    st.markdown("### Analysis")
    df = st.session_state.store_linearity
    if not df.empty and "Target Conc (mg/L)" in df.columns and "Absorbance" in df.columns:
        x = pd.to_numeric(df["Target Conc (mg/L)"], errors="coerce")
        y = pd.to_numeric(df["Absorbance"], errors="coerce")
        mask = x.notna() & y.notna()
        # weights by inverse variance of replicate groups
        grouped = df[mask].groupby("Target Conc (mg/L)")["Absorbance"].agg(["mean","std","count"])
        weights = None
        if (grouped["std"]>0).any():
            w_map = (1.0/(grouped["std"]**2)).to_dict()
            weights = np.array([w_map.get(v, 1.0) for v in df.loc[mask, "Target Conc (mg/L)"]])
        fit = fit_linear(x[mask], y[mask], weights=weights)
        if fit:
            st.write(f"**Slope:** {fit['slope']:.6f}  |  **Intercept:** {fit['intercept']:.6f}  |  **R²:** {fit['R2']:.5f}")
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(fit["x"], fit["y"])
            ax.plot(fit["x"], fit["yhat"])
            ax.set_xlabel("Conc (mg/L)"); ax.set_ylabel("Absorbance (A)"); ax.set_title("Calibration")
            st.pyplot(fig)
            save_fig_download(fig, "linearity_calibration")

            # Residuals
            res = fit["y"] - fit["yhat"]
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.scatter(fit["x"], res)
            ax2.axhline(0)
            ax2.set_xlabel("Conc (mg/L)"); ax2.set_ylabel("Residuals"); ax2.set_title("Residual Plot")
            st.pyplot(fig2)
            save_fig_download(fig2, "linearity_residuals")

            # Prepared model equation for device
            st.info(f"Calibration equation (A = m·C + b):  m={fit['slope']:.6f},  b={fit['intercept']:.6f}")
        else:
            st.warning("Not enough valid points to fit a model.")

# =============================
# SECTION 2: INTERFERENCE
# =============================
elif section.startswith("2"):
    st.subheader("2) Interference Study – DOE → Data → Analysis")
    colA, colB = st.columns(2)
    with colA:
        interfs = st.text_input("Interferents (comma-separated ions/chemicals)", "Cu2+,Ni2+,Co2+,Zn2+,Ca2+,Mg2+,Chloride,Sulfate")
        interferents = [s.strip() for s in interfs.split(",") if s.strip()]
        ilv = st.text_input("Interference Levels (e.g., Low,Medium,High)", "Low,Medium,High")
        interf_levels = [s.strip() for s in ilv.split(",") if s.strip()]
    with colB:
        fe_levels_txt = st.text_input("Fe Levels (mg/L)", "5,10,20")
        fe_levels = [float(x) for x in fe_levels_txt.split(",")]
        reps = st.number_input("Replicates per cell", 1, 10, 1)

    if st.button("Generate DOE", key="gen_int"):
        st.session_state.store_interference = doe_interference(interferents, interf_levels, fe_levels, reps)

    st.dataframe(st.session_state.store_interference, use_container_width=True, height=360)

    st.markdown("### Analysis (Recovery vs expected)")
    df = st.session_state.store_interference
    if not df.empty and "Fe Level (mg/L)" in df.columns and "Absorbance" in df.columns:
        st.caption("Provide expected Abs from calibration if available to compute recovery.")
        # Basic group stats by interferent/level
        stats_df = precision_stats(df.rename(columns={"Fe Level (mg/L)":"Conc (mg/L)"}),
                                   conc_col="Conc (mg/L)", abs_col="Absorbance",
                                   group_cols=["Interferent","Interf. Level","Conc (mg/L)"])
        st.dataframe(stats_df, use_container_width=True)

# =============================
# SECTION 3: REPEATABILITY
# =============================
elif section.startswith("3"):
    st.subheader("3) Repeatability (Intra-day Precision)")
    levels_txt = st.text_input("Concentration levels (mg/L)", "5,12.5,20")
    levels = [float(x) for x in levels_txt.split(",")]
    reps = st.number_input("Replicates per level", 2, 50, 10)
    if st.button("Generate DOE", key="gen_rep"):
        st.session_state.store_repeatability = doe_repeatability(levels, reps)
    st.dataframe(st.session_state.store_repeatability, use_container_width=True, height=360)

    st.markdown("### Analysis")
    df = st.session_state.store_repeatability
    if not df.empty and "Conc (mg/L)" in df.columns and "Absorbance" in df.columns:
        res = precision_stats(df, conc_col="Conc (mg/L)", abs_col="Absorbance")
        st.dataframe(res, use_container_width=True)

# =============================
# SECTION 4: INTERMEDIATE PRECISION
# =============================
elif section.startswith("4"):
    st.subheader("4) Intermediate Precision (Inter-day)")
    levels_txt = st.text_input("Concentration levels (mg/L)", "5,12.5,20")
    levels = [float(x) for x in levels_txt.split(",")]
    days = st.number_input("Days", 2, 30, 5)
    reps_per_day = st.number_input("Replicates per day", 1, 20, 3)
    if st.button("Generate DOE", key="gen_ip"):
        st.session_state.store_intermediate = doe_intermediate(levels, days, reps_per_day)
    st.dataframe(st.session_state.store_intermediate, use_container_width=True, height=360)

    st.markdown("### Analysis")
    df = st.session_state.store_intermediate
    if not df.empty and "Conc (mg/L)" in df.columns and "Absorbance" in df.columns:
        res = precision_stats(df, conc_col="Conc (mg/L)", abs_col="Absorbance", group_cols=["Conc (mg/L)","Day"])
        st.dataframe(res, use_container_width=True)

# =============================
# SECTION 5: ACCURACY / RECOVERY
# =============================
elif section.startswith("5"):
    st.subheader("5) Accuracy / Recovery")
    matrices = st.text_input("Matrices (comma-separated)", "DI Water,Tap Water,River Water")
    matrices = [s.strip() for s in matrices.split(",") if s.strip()]
    spike_txt = st.text_input("Spike levels (mg/L)", "5,10,15,20")
    spikes = [float(x) for x in spike_txt.split(",")]
    reps = st.number_input("Replicates", 1, 10, 3)
    if st.button("Generate DOE", key="gen_acc"):
        st.session_state.store_accuracy = doe_accuracy(matrices, spikes, reps)
    st.dataframe(st.session_state.store_accuracy, use_container_width=True, height=360)

    st.markdown("### Analysis")
    df = st.session_state.store_accuracy
    if not df.empty and "Measured (mg/L)" in df.columns and "Expected (mg/L)" in df.columns:
        df["Recovery %"] = pd.to_numeric(df["Measured (mg/L)"], errors="coerce") / pd.to_numeric(df["Expected (mg/L)"], errors="coerce") * 100.0
        st.dataframe(accuracy_stats(df), use_container_width=True)

# =============================
# SECTION 6: LOD / LOQ
# =============================
elif section.startswith("6"):
    st.subheader("6) LOD / LOQ")
    blk = st.number_input("Blank replicates (n)", 5, 30, 10)
    low_txt = st.text_input("Low levels (mg/L)", "0.5,1.0,1.5")
    lows = [float(x) for x in low_txt.split(",")]
    reps = st.number_input("Replicates per low level", 1, 10, 3)
    if st.button("Generate DOE", key="gen_lod"):
        st.session_state.store_lodloq = doe_lodloq(blk, lows, reps)
    st.dataframe(st.session_state.store_lodloq, use_container_width=True, height=360)

    st.markdown("### Analysis (blank-based)")
    # Need slope from linearity
    st.caption("Provide slope from the current linearity calibration (A = m·C + b).")
    slope_in = st.number_input("Slope (m)", step=1e-6, format="%.6f")
    df = st.session_state.store_lodloq
    if slope_in and not df.empty and "Type" in df.columns and "Absorbance" in df.columns:
        blanks = pd.to_numeric(df.loc[df["Type"]=="Blank","Absorbance"], errors="coerce").dropna()
        sd_b, lod, loq = lod_loq_from_blanks(blanks, slope_in)
        st.write(f"Blank SD = **{sd_b:.6f} A**")
        st.write(f"LOD (3σ/m) = **{lod:.6f} mg/L**")
        st.write(f"LOQ (10σ/m) = **{loq:.6f} mg/L**")

# =============================
# SECTION 7: STABILITY
# =============================
elif section.startswith("7"):
    st.subheader("7) Stability")
    levels_txt = st.text_input("Concentration levels (mg/L)", "5,12.5,20")
    levels = [float(x) for x in levels_txt.split(",")]
    t_txt = st.text_input("Timepoints (min)", "0,15,30,60,120,180,240,300")
    tps = [float(x) for x in t_txt.split(",")]
    reps = st.number_input("Replicates per timepoint", 1, 5, 2)
    if st.button("Generate DOE", key="gen_stab"):
        st.session_state.store_stability = doe_stability(levels, tps, reps)
    st.dataframe(st.session_state.store_stability, use_container_width=True, height=360)

    st.markdown("### Analysis")
    df = st.session_state.store_stability
    if not df.empty and "Time (min)" in df.columns and "Absorbance" in df.columns:
        # % change relative to t=0 per level
        out = []
        for c, sub in df.groupby("Conc (mg/L)"):
            sub_sorted = sub.sort_values("Time (min)")
            t0 = pd.to_numeric(sub_sorted.loc[sub_sorted["Time (min)"]==0,"Absorbance"], errors="coerce").mean()
            for _, row in sub_sorted.iterrows():
                a = pd.to_numeric(pd.Series([row["Absorbance"]]), errors="coerce").iloc[0]
                if pd.notna(a) and pd.notna(t0) and t0!=0:
                    out.append([c, row["Time (min)"], (a-t0)/t0*100.0])
        if out:
            s = pd.DataFrame(out, columns=["Conc (mg/L)","Time (min)","% Change"])
            st.dataframe(s.groupby(["Conc (mg/L)","Time (min)"])["% Change"].mean().reset_index(), use_container_width=True)

# =============================
# SECTION 8: ROBUSTNESS
# =============================
elif section.startswith("8"):
    st.subheader("8) Robustness")
    st.caption("Test small, deliberate variations around nominal conditions.")
    default_factors = [
        ("pH", "3.5", "±0.2"),
        ("Reaction Time", "15 min", "±2 min"),
        ("Temperature", "25°C", "±5°C"),
        ("Reagent Age", "Fresh", "1 week"),
        ("Sample Volume", "10 mL", "±1 mL"),
        ("Buffer Volume", "2 mL", "±0.2 mL"),
        ("Complexant Conc", "0.1%", "±0.02%"),
        ("Reducer", "10%", "±2%"),
    ]
    if st.button("Generate DOE", key="gen_rob"):
        st.session_state.store_robustness = doe_robustness(default_factors, test_conc=12.5, tests_per_factor=3)
    st.dataframe(st.session_state.store_robustness, use_container_width=True, height=360)

# =============================
# SECTION 9: SAMPLE MATRIX EFFECTS
# =============================
elif section.startswith("9"):
    st.subheader("9) Sample Matrix Effects")
    mats = st.text_input("Matrices", "Tap Water,River Water,Groundwater")
    matrices = [s.strip() for s in mats.split(",") if s.strip()]
    levels_txt = st.text_input("Concentration levels (mg/L)", "5,10,15,20")
    levels = [float(x) for x in levels_txt.split(",")]
    reps = st.number_input("Replicates per cell", 1, 10, 3)
    if st.button("Generate DOE", key="gen_mat"):
        st.session_state.store_matrix = doe_matrix(matrices, levels, reps)
    st.dataframe(st.session_state.store_matrix, use_container_width=True, height=360)

# =============================
# PROJECT & SETTINGS
# =============================
else:
    st.subheader("Project & Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Export Project (ZIP)**")
        def export_zip():
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("loc_config.csv", st.session_state.loc_config.to_csv(index=False))
                z.writestr("vol_config.json", json.dumps(st.session_state.vol_config, indent=2))
                for key in [
                    "store_linearity","store_interference","store_repeatability",
                    "store_intermediate","store_accuracy","store_lodloq",
                    "store_stability","store_robustness","store_matrix"
                ]:
                    df = st.session_state.get(key, pd.DataFrame())
                    z.writestr(f"{key}.csv", df.to_csv(index=False))
                z.writestr("raw_log.csv", st.session_state.raw_log.to_csv(index=False))
            buf.seek(0)
            return buf.getvalue()
        st.download_button("Download Project ZIP", data=export_zip(), file_name="rd_md_analysis_project.zip", mime="application/zip")
    with col2:
        st.markdown("**Import Project (ZIP)**")
        upz = st.file_uploader("Upload ZIP", type=["zip"], key="proj_zip")
        if upz:
            try:
                zf = zipfile.ZipFile(io.BytesIO(upz.read()))
                if "loc_config.csv" in zf.namelist():
                    st.session_state.loc_config = pd.read_csv(zf.open("loc_config.csv"))
                if "vol_config.json" in zf.namelist():
                    st.session_state.vol_config = json.loads(zf.read("vol_config.json"))
                for key in [
                    "store_linearity","store_interference","store_repeatability",
                    "store_intermediate","store_accuracy","store_lodloq",
                    "store_stability","store_robustness","store_matrix"
                ]:
                    nm = f"{key}.csv"
                    if nm in zf.namelist():
                        st.session_state[key] = pd.read_csv(zf.open(nm))
                if "raw_log.csv" in zf.namelist():
                    st.session_state.raw_log = pd.read_csv(zf.open("raw_log.csv"))
                st.success("Project restored.")
            except Exception as e:
                st.error(f"Failed to import project: {e}")

    st.markdown("---")
    st.subheader("Raw Import Log")
    st.dataframe(st.session_state.raw_log, use_container_width=True, height=360)

st.markdown("---")
st.caption("Note: The app expects device JSON files with both Background and Sample scans and an LED channel (default: SC_Green). Concentration from spike is auto-computed via M₁V₁ = M₂V₂ using LOC configuration.")
