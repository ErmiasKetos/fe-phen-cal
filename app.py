
# R&D MD and Analysis – Universal Colorimetric Method Development App
# (Streamlit)
#
# Features:
# - Project Settings (sample volume, LED channel, include LOC volumes, LOC config w/ stock conc)
# - JSON ingestion (Background/Sample, robust mean, absorbance A = log10(BG/S))
# - Auto concentration via M1V1=M2V2 using LOC standard from config
# - Linearity: DOE generator (manual or auto), variance-weighted regression (1/SD^2), PNG/PDF export
# - Placeholder pages for Interference, Repeatability, Intermediate Precision, Accuracy,
#   LOD/LOQ, Stability, Robustness, Sample Matrix with working dataframes and CSV export
# - Safe linear solver with centering/scaling, weight clipping, ridge fallback, weighted R^2
#
# Usage:
#   pip install -r requirements.txt
#   streamlit run rd_md_analysis_app.py

import io
import json
import math
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------- Utilities -------------

def robust_mean(values: List[float]) -> float:
    """MAD-based outlier rejection then mean."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        return float(np.mean(arr))
    thresh = 3 * 1.4826 * mad
    filt = arr[np.abs(arr - med) <= thresh]
    if filt.size == 0:
        filt = arr
    return float(np.mean(filt))

def get_nested(obj: Any, *keys, default=None):
    cur = obj
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        else:
            return default
    return default if cur is None else cur

def find_scans(obj: Any) -> List[dict]:
    """Recursively find a list-like 'scans' node."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() == "scans" and isinstance(v, list):
                return v
            res = find_scans(v)
            if res:
                return res
    elif isinstance(obj, list):
        for v in obj:
            res = find_scans(v)
            if res:
                return res
    return []

def scan_type(scan: dict) -> str:
    # many places to store
    t = (scan.get("scanType")
         or get_nested(scan, "parameters", "scanType")
         or get_nested(scan, "parameters", "scan_type")
         or "").lower()
    if "back" in t or "bg" in t:
        return "background"
    if "sam" in t:
        return "sample"
    # Also check name/type direct
    t2 = (scan.get("type") or scan.get("name") or "").lower()
    if "back" in t2 or "bg" in t2:
        return "background"
    if "sam" in t2:
        return "sample"
    return ""

def get_led_array(scan: dict, channel_key: str) -> List[float]:
    """Find array for LED channel (default SC_Green)."""
    if not isinstance(scan, dict):
        return []
    # direct
    if channel_key in scan:
        return parse_values(scan[channel_key])
    # nested containers
    for key in ("channels", "Channels", "data", "Data"):
        node = scan.get(key)
        if isinstance(node, dict):
            # exact key
            if channel_key in node:
                return parse_values(node[channel_key])
            # case-insensitive
            for k, v in node.items():
                if k.lower() == channel_key.lower():
                    return parse_values(v)
    # last resort: search values
    for k, v in scan.items():
        if isinstance(v, list) and all(isinstance(x, (int, float, str, dict)) for x in v):
            vals = parse_values(v)
            if len(vals) >= 5:
                return vals
    return []

def parse_values(raw) -> List[float]:
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
                for key in ("value", "intensity"):
                    if key in v:
                        try:
                            out.append(float(v[key]))
                        except:
                            pass
        return [x for x in out if np.isfinite(x)]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except:
                pass
        return [x for x in out if np.isfinite(x)]
    return []

def extract_loc_doses(scan: dict) -> Dict[str, float]:
    doses = {}
    if not isinstance(scan, dict):
        return doses
    # scan-level
    for src in (scan, scan.get("parameters", {}) or {}):
        for k, v in list(src.items()):
            if isinstance(k, str) and k.upper().startswith("LOC"):
                try:
                    val = float(v)
                except:
                    continue
                if val > 0:
                    doses[k.upper()] = val
    return doses

def pick_bg_and_last_sample(scans: List[dict]) -> Tuple[dict, dict]:
    bg = None
    last_sample = None
    for s in scans:
        t = scan_type(s)
        if t == "background" and bg is None:
            bg = s
        elif t == "sample":
            last_sample = s
    return bg, last_sample

# ------------- Regression -------------

def variance_weights_from_replicates(xs, ys, group_ids=None, sd_floor=None):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if ys.size == 0:
        return np.array([])
    if sd_floor is None:
        y_rng = (np.nanmax(ys) - np.nanmin(ys)) if np.isfinite(ys).any() else 1.0
        sd_floor = max(1e-6, 0.01 * y_rng)
    if group_ids is None:
        gids = np.round(xs, 12)
    else:
        gids = np.asarray(group_ids)

    weights = np.empty_like(ys, dtype=float)
    for g in np.unique(gids):
        m = (gids == g)
        if np.sum(m) < 2:
            sd = sd_floor
        else:
            sd = np.nanstd(ys[m], ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                sd = sd_floor
        w = 1.0 / (max(sd, sd_floor) ** 2)
        weights[m] = w
    return weights

def fit_linear(x, y, weights=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        mask &= np.isfinite(w)
        w = w[mask]
    x = x[mask]; y = y[mask]
    if x.size < 2 or np.unique(np.round(x,12)).size < 2:
        raise ValueError("Need at least two distinct x values for a line.")
    x_mean = x.mean()
    x_std = x.std()
    if x_std == 0:
        raise ValueError("Zero variance in x.")
    xs = (x - x_mean) / x_std
    X = np.c_[np.ones_like(xs), xs]

    if weights is None:
        W = np.ones_like(xs)
    else:
        w = np.asarray(weights, dtype=float)[mask]
        w = np.where(w > 0, w, np.nan)
        if np.isnan(w).any():
            pos = w[np.isfinite(w)]
            floor = np.nanpercentile(pos, 5) if pos.size else 1.0
            w = np.where(np.isfinite(w), w, floor)
        lo = np.nanpercentile(w, 1)
        hi = np.nanpercentile(w, 99)
        if not np.isfinite(hi) or hi <= 0: hi = 1.0
        if not np.isfinite(lo) or lo <= 0: lo = hi * 1e-6
        w = np.clip(w, lo, hi)
        W = np.sqrt(w)

    try:
        beta_scaled, *_ = np.linalg.lstsq(W[:,None]*X, W*y, rcond=None)
    except np.linalg.LinAlgError:
        XtWX = (X.T * (W**2)) @ X
        XtWy = X.T @ (W**2 * y)
        lam = 1e-8 * np.trace(XtWX) if np.isfinite(np.trace(XtWX)) else 1e-6
        beta_scaled = np.linalg.solve(XtWX + lam*np.eye(2), XtWy)

    b0_s, b1_s = beta_scaled
    slope = b1_s / x_std
    intercept = b0_s - slope * x_mean

    yhat = intercept + slope * x
    if weights is None:
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
    else:
        ybar = np.average(y, weights=w)
        ss_res = np.sum(w * (y - yhat)**2)
        ss_tot = np.sum(w * (y - ybar)**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2

def export_plot(fig, png_name="plot.png", pdf_name="plot.pdf"):
    # returns (png_bytes, pdf_bytes)
    pbuf = io.BytesIO()
    fig.savefig(pbuf, format="png", bbox_inches="tight", dpi=200)
    pbuf.seek(0)
    dbuf = io.BytesIO()
    fig.savefig(dbuf, format="pdf", bbox_inches="tight")
    dbuf.seek(0)
    return pbuf.getvalue(), dbuf.getvalue()

# ------------- Streamlit App -------------

st.set_page_config(page_title="R&D MD and Analysis", layout="wide")

if "settings" not in st.session_state:
    st.session_state.settings = {
        "analyte": "Iron (Fe)",
        "led_channel": "SC_Green",
        "base_sample_mL": 40.0,
        "extra_const_mL": 0.0,
        "include_loc_vols": True,
        "standard_loc": "LOC15",
        "standard_stock_mgL": 1000.0,
        "loc_map": {f"LOC{i}": "" for i in range(1,17)},
    }

if "raw_log" not in st.session_state:
    st.session_state.raw_log = []  # list of dict rows

if "linearity" not in st.session_state:
    st.session_state.linearity = pd.DataFrame(columns=[
        "Test #","Shield Test #","Date","Conc (mg/L)","LOC Dosing (µL)",
        "Replicate","Absorbance","BG Mean","Sample Mean","Temp (°C)","Status","Notes"
    ])

def sidebar():
    st.sidebar.title("R&D MD & Analysis")
    page = st.sidebar.radio("Go to", [
        "Project & Settings",
        "Linearity",
        "Interference Study",
        "Repeatability (Intra-day)",
        "Intermediate Precision",
        "Accuracy/Recovery",
        "LOD/LOQ",
        "Stability",
        "Robustness",
        "Sample Matrix Effects",
        "Raw Import Log / Export"
    ])
    return page

page = sidebar()

# ----- Project & Settings -----
if page == "Project & Settings":
    st.header("Project & Settings")
    s = st.session_state.settings

    col1, col2 = st.columns(2)
    with col1:
        s["analyte"] = st.text_input("Analyte name", s["analyte"])
        s["led_channel"] = st.text_input("LED channel key", s["led_channel"])
        s["base_sample_mL"] = st.number_input("Base sample volume (mL)", 1.0, 200.0, s["base_sample_mL"], 0.1)
        s["extra_const_mL"] = st.number_input("Extra constant volume (mL)", 0.0, 50.0, s["extra_const_mL"], 0.1)
        s["include_loc_vols"] = st.checkbox("Include LOC dosing volumes into total volume", value=s["include_loc_vols"])
    with col2:
        st.markdown("**Standard configuration (for auto-concentration)**")
        s["standard_loc"] = st.selectbox("Standard LOC", [f"LOC{i}" for i in range(1,17)], index=14)  # default LOC15
        s["standard_stock_mgL"] = st.number_input("Standard stock concentration (mg/L)", 0.0, 100000.0, s["standard_stock_mgL"], 1.0)

    st.markdown("---")
    st.subheader("LOC Reagent Map (optional docs)")
    loc_df = pd.DataFrame({"LOC":[f"LOC{i}" for i in range(1,17)],
                           "Role":[""]*16,
                           "Notes":[""]*16})
    st.dataframe(loc_df, use_container_width=True)

    st.success("Settings saved in session.")

# ----- JSON Import helper -----

def concentration_from_loc_doses(loc_doses: Dict[str, float], settings: dict) -> float:
    """Compute C_final from standard LOC spike and total volume (mL)."""
    std_loc = settings["standard_loc"]
    stock = float(settings["standard_stock_mgL"] or 0.0)
    base = float(settings["base_sample_mL"] or 0.0)
    extra = float(settings["extra_const_mL"] or 0.0)
    include = bool(settings["include_loc_vols"])

    spike_uL = float(loc_doses.get(std_loc, 0.0))
    if spike_uL <= 0 or stock <= 0 or base <= 0:
        return 0.0
    total_mL = base + extra
    if include:
        total_mL += (sum(loc_doses.values())/1000.0)
    c_final = stock * ((spike_uL/1000.0)/total_mL)
    return float(c_final)

def parse_device_json(file_bytes: bytes, settings: dict) -> dict:
    js = json.loads(file_bytes.decode("utf-8"))
    scans = find_scans(js)
    bg, sample = pick_bg_and_last_sample(scans)
    led = settings["led_channel"]

    bg_vals = get_led_array(bg, led) if bg else []
    sm_vals = get_led_array(sample, led) if sample else []
    bg_mean = robust_mean(bg_vals) if bg_vals else np.nan
    sm_mean = robust_mean(sm_vals) if sm_vals else np.nan
    A = np.nan
    if np.isfinite(bg_mean) and np.isfinite(sm_mean) and sm_mean > 0:
        A = math.log10(bg_mean/sm_mean)

    loc = extract_loc_doses(sample) if sample else {}
    conc = concentration_from_loc_doses(loc, settings)
    temp = None
    for key in ("bb_temp","temperature","temp"):
        t = get_nested(sample or {}, key) or get_nested(js, "payload", key)
        if t is not None:
            try: 
                temp = float(t); 
                break
            except: 
                pass

    # Attempt several shield test fields
    shield = (get_nested(js, "payload", "exp_number")
              or get_nested(js, "payload", "test_number")
              or js.get("exp_number") or js.get("test_number")
              or js.get("testNumber") or js.get("shield_test_number") or "")
    return {
        "Shield Test #": str(shield),
        "Conc (mg/L)": float(conc),
        "LOC Dosing (µL)": ", ".join([f"{k}:{v:g}" for k,v in loc.items()]) if loc else "",
        "Absorbance": float(A) if np.isfinite(A) else np.nan,
        "BG Mean": float(bg_mean) if np.isfinite(bg_mean) else np.nan,
        "Sample Mean": float(sm_mean) if np.isfinite(sm_mean) else np.nan,
        "Temp (°C)": temp,
    }

def add_to_raw_log(row: dict, section: str, status: str, filename: str):
    st.session_state.raw_log.append({
        "Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Section": section,
        "File": filename,
        **row,
        "Status": status
    })

# ----- Linearity -----
if page == "Linearity":
    st.header("Linearity – DOE, JSON Import, Fit, and Exports")

    # DOE Builder
    with st.expander("Design of Experiments (DOE)", expanded=True):
        mode = st.radio("Levels input mode", ["Auto-generate", "Manual"], horizontal=True)
        if mode == "Auto-generate":
            lo = st.number_input("Range start (mg/L)", 0.0, 1e6, 0.0, 0.1)
            hi = st.number_input("Range end (mg/L)", 0.0, 1e6, 25.0, 0.1)
            points = st.selectbox("Number of calibration levels", [5,6,7,8,9,10,11,12], index=7)
            replicates = st.number_input("Replicates per level", 1, 10, 3)
            levels = list(np.linspace(lo, hi, points)) if hi >= lo else []
        else:
            levels_str = st.text_input("Levels (mg/L, comma-separated)", "0,1,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25")
            try:
                levels = [float(x.strip()) for x in levels_str.split(",") if x.strip()!=""]
            except:
                st.error("Invalid manual levels.")
                levels = []
            replicates = st.number_input("Replicates per level", 1, 10, 3)

        if st.button("Generate DOE Table"):
            rows = []
            tnum = 1
            for c in levels:
                for r in range(1, int(replicates)+1):
                    rows.append([f"L-{tnum:03d}","", "", float(c), "", r, "", "", "", "", "PENDING", ""])
                    tnum+=1
            st.session_state.linearity = pd.DataFrame(rows, columns=st.session_state.linearity.columns)

    st.markdown("#### DOE Table")
    st.dataframe(st.session_state.linearity, use_container_width=True)

    # JSON import
    st.markdown("---")
    st.subheader("Import device JSON → auto Absorbance & Conc")
    up = st.file_uploader("Upload one or more device JSON files", type=["json"], accept_multiple_files=True)
    if up:
        for f in up:
            try:
                row = parse_device_json(f.read(), st.session_state.settings)
                # place into first empty matching row by concentration (if any)
                df = st.session_state.linearity
                # find candidate rows where conc matches and Absorbance empty
                idx = df[(np.isclose(df["Conc (mg/L)"], row["Conc (mg/L)"], rtol=0, atol=1e-9)) & (df["Absorbance"].isna() | (df["Absorbance"]==""))].index
                target = idx[0] if len(idx)>0 else None
                if target is None:
                    # append as extra row
                    new = {
                        "Test #": f"IMP-{len(df)+1:03d}",
                        "Shield Test #": row["Shield Test #"],
                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Conc (mg/L)": row["Conc (mg/L)"],
                        "LOC Dosing (µL)": row["LOC Dosing (µL)"],
                        "Replicate": 1,
                        "Absorbance": row["Absorbance"],
                        "BG Mean": row["BG Mean"],
                        "Sample Mean": row["Sample Mean"],
                        "Temp (°C)": row["Temp (°C)"],
                        "Status": "COMPLETE",
                        "Notes": "Auto-inserted"
                    }
                    st.session_state.linearity = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                else:
                    st.session_state.linearity.loc[target, ["Shield Test #","Date","LOC Dosing (µL)","Absorbance","BG Mean","Sample Mean","Temp (°C)","Status"]] = [
                        row["Shield Test #"],
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        row["LOC Dosing (µL)"],
                        row["Absorbance"],
                        row["BG Mean"],
                        row["Sample Mean"],
                        row["Temp (°C)"],
                        "COMPLETE"
                    ]
                add_to_raw_log(row, "Linearity", "INSERTED", f.name)
                st.success(f"Imported {f.name}")
            except Exception as e:
                add_to_raw_log({"Error": str(e)}, "Linearity", "FAILED", f.name)
                st.error(f"Failed to import {f.name}: {e}")

    # Fit
    st.markdown("---")
    st.subheader("Calibration Fit")
    df = st.session_state.linearity.copy()
    # drop rows without absorbance or conc
    df = df[pd.to_numeric(df["Conc (mg/L)"], errors="coerce").notna() & pd.to_numeric(df["Absorbance"], errors="coerce").notna()]
    if len(df) >= 2 and df["Conc (mg/L)"].nunique() >= 2:
        colA, colB = st.columns(2)
        with colA:
            use_weights = st.checkbox("Variance-weighted (1/SD² by level)", value=True)
        with colB:
            show_resid = st.checkbox("Show residuals table", value=False)

        # build x,y and (optional) weights from replicate groups
        x = df["Conc (mg/L)"].astype(float).values
        y = df["Absorbance"].astype(float).values
        weights = None
        if use_weights:
            # group by concentration
            weights = variance_weights_from_replicates(x, y, group_ids=np.round(x, 12))

        try:
            m, b, r2 = fit_linear(x, y, weights=weights)
            st.info(f"**Calibration**: A = m·C + b  →  m = {m:.6g},  b = {b:.6g},  R² = {r2:.6f}")
            # plot
            fig = plt.figure()
            ax = plt.gca()
            ax.scatter(x, y)
            xs = np.linspace(min(x), max(x), 100)
            ax.plot(xs, m*xs + b)
            ax.set_xlabel("Concentration (mg/L)")
            ax.set_ylabel("Absorbance (A)")
            ax.set_title("Calibration Curve")
            st.pyplot(fig)

            png, pdf = export_plot(fig, "calibration.png", "calibration.pdf")
            st.download_button("Download plot (PNG)", data=png, file_name="calibration.png", mime="image/png")
            st.download_button("Download plot (PDF)", data=pdf, file_name="calibration.pdf", mime="application/pdf")

            if show_resid:
                res = pd.DataFrame({"Conc (mg/L)": x, "Absorbance": y, "Fit": m*x + b})
                res["Residual"] = res["Absorbance"] - res["Fit"]
                st.dataframe(res.sort_values("Conc (mg/L)"))
        except Exception as e:
            st.error(f"Fit failed: {e}")
    else:
        st.warning("Add at least two distinct concentration levels with absorbance to fit.")

# ----- Placeholder, working tables for other studies -----

def generic_table_page(title: str, key: str, columns: List[str]):
    st.header(title)
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame(columns=columns)
    st.markdown("Upload device JSON files to populate Absorbance/Conc quickly, or paste/edit rows manually.")

    up = st.file_uploader("Upload JSON (optional)", type=["json"], accept_multiple_files=True, key=f"up_{key}")
    if up:
        for f in up:
            try:
                row = parse_device_json(f.read(), st.session_state.settings)
                new = {
                    "Shield Test #": row["Shield Test #"],
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Conc (mg/L)": row["Conc (mg/L)"],
                    "Absorbance": row["Absorbance"],
                    "BG Mean": row["BG Mean"],
                    "Sample Mean": row["Sample Mean"],
                    "Temp (°C)": row["Temp (°C)"],
                    "Notes": "Auto-import"
                }
                st.session_state[key] = pd.concat([st.session_state[key], pd.DataFrame([new])], ignore_index=True)
                add_to_raw_log(row, title, "INSERTED", f.name)
            except Exception as e:
                add_to_raw_log({"Error": str(e)}, title, "FAILED", f.name)
                st.error(f"Failed to import {f.name}: {e}")

    st.dataframe(st.session_state[key], use_container_width=True)
    csv = st.session_state[key].to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", data=csv, file_name=f"{key}.csv", mime="text/csv")

if page == "Interference Study":
    cols = ["Shield Test #","Date","Interferent","Interf. Level","Conc (mg/L)","Absorbance","BG Mean","Sample Mean","Temp (°C)","Notes"]
    generic_table_page("Interference Study", "interference", cols)

if page == "Repeatability (Intra-day)":
    cols = ["Shield Test #","Date","Conc (mg/L)","Replicate","Absorbance","BG Mean","Sample Mean","Temp (°C)","Notes"]
    generic_table_page("Repeatability (Intra-day)", "repeatability", cols)

if page == "Intermediate Precision":
    cols = ["Shield Test #","Day","Date","Analyst","Conc (mg/L)","Replicate","Absorbance","BG Mean","Sample Mean","Temp (°C)","Notes"]
    generic_table_page("Intermediate Precision", "intermediate_precision", cols)

if page == "Accuracy/Recovery":
    cols = ["Shield Test #","Date","Matrix","Spike (mg/L)","Replicate","Measured (mg/L)","Expected (mg/L)","Recovery %","Absorbance","Notes"]
    generic_table_page("Accuracy/Recovery", "accuracy", cols)

if page == "LOD/LOQ":
    st.header("LOD/LOQ")
    st.markdown("Provide blank absorbances and reference slope (from linearity) to compute LOD/LOQ.")
    if "lod" not in st.session_state:
        st.session_state.lod = pd.DataFrame(columns=["Test #","Date","Absorbance","Notes"])
    # blanks table
    st.subheader("Blank measurements")
    st.session_state.lod = st.dataframe(st.session_state.lod, use_container_width=True).data

    slope = st.number_input("Slope m (from linearity)", value=1.0, step=0.0001, format="%.6f")
    # compute stats if blanks present
    try:
        blks = pd.to_numeric(st.session_state.lod["Absorbance"], errors="coerce").dropna().values
        if blks.size >= 3 and slope>0:
            sd_b = float(np.std(blks, ddof=1)) if blks.size>1 else float(np.std(blks))
            lod = 3*sd_b/slope
            loq = 10*sd_b/slope
            st.info(f"Blank SD = {sd_b:.6g} → LOD = {lod:.6g} mg/L, LOQ = {loq:.6g} mg/L")
        else:
            st.warning("Enter ≥3 blank absorbances and a positive slope to compute LOD/LOQ.")
    except Exception as e:
        st.error(f"LOD/LOQ calc error: {e}")

if page == "Stability":
    cols = ["Shield Test #","Prep Date","Measure Time","Elapsed (min)","Conc (mg/L)","Replicate","Absorbance","Initial Abs","% Change","Notes"]
    generic_table_page("Stability", "stability", cols)

if page == "Robustness":
    cols = ["Shield Test #","Date","Factor","Nominal","Test Value","Conc (mg/L)","Replicate","Absorbance","% Difference","Notes"]
    generic_table_page("Robustness", "robustness", cols)

if page == "Sample Matrix Effects":
    cols = ["Shield Test #","Date","Matrix","Matrix ID","Conc (mg/L)","Replicate","Absorbance","DI Water Abs","Matrix Effect %","Notes"]
    generic_table_page("Sample Matrix Effects", "matrix", cols)

if page == "Raw Import Log / Export":
    st.header("Raw Import Log / Export")
    if len(st.session_state.raw_log)==0:
        st.info("No imports yet.")
    else:
        df = pd.DataFrame(st.session_state.raw_log)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Import Log (CSV)", data=csv, file_name="import_log.csv", mime="text/csv")

        # Export a project bundle
        st.subheader("Project Export (.zip)")
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("settings.json", json.dumps(st.session_state.settings, indent=2))
            zf.writestr("linearity.csv", st.session_state.linearity.to_csv(index=False))
            for key in ["interference","repeatability","intermediate_precision","accuracy","lod","stability","robustness","matrix"]:
                if key in st.session_state:
                    zf.writestr(f"{key}.csv", st.session_state[key].to_csv(index=False))
            if len(st.session_state.raw_log):
                zf.writestr("import_log.csv", pd.DataFrame(st.session_state.raw_log).to_csv(index=False))
        zbuf.seek(0)
        st.download_button("Download Project ZIP", data=zbuf.getvalue(), file_name="rd_md_project.zip", mime="application/zip")
