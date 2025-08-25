#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
import json
import math
import statistics as stats
from datetime import datetime
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fe Phenanthroline — Method Development & Analysis", layout="wide")

# =========================
# Utility functions
# =========================

def parse_values(raw):
    """
    Accepts either a list of numbers, a comma-separated string of numbers,
    or a list of dicts with 'value'/'intensity'. Returns list[float].
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for item in raw:
            try:
                out.append(float(item))
            except Exception:
                if isinstance(item, dict):
                    if "value" in item:
                        out.append(float(item["value"]))
                    elif "intensity" in item:
                        out.append(float(item["intensity"]))
        return out
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                return [float(x) for x in arr]
            except Exception:
                pass
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except Exception:
            return []
    return []

def robust_mean(values):
    """
    Median/MAD outlier filter (3*MAD). Returns (mean_of_kept, kept_values).
    """
    vals = [float(v) for v in values if v is not None]
    if len(vals) == 0:
        return float("nan"), []
    med = stats.median(vals)
    mad = stats.median([abs(x - med) for x in vals])
    if mad == 0:
        return sum(vals)/len(vals), vals
    thr = 3.0 * 1.4826 * mad
    kept = [x for x in vals if abs(x - med) <= thr]
    if not kept:
        kept = vals
    return sum(kept)/len(kept), kept

def _find_scans(obj):
    """
    Recursively search the JSON object for a key named 'scans' that holds a list.
    Returns the first such list found or None.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "scans" and isinstance(v, list):
                return v
            found = _find_scans(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_scans(item)
            if found is not None:
                return found
    return None

def _get_channel_array(node, led_key):
    """
    Extract the array for the given LED from a scan node.
    Accepts direct key, case-insensitive key, or nested under 'channels'/'Channels'/'data'.
    """
    if led_key in node:
        return parse_values(node.get(led_key))
    for k in list(node.keys()):
        if isinstance(k, str) and k.lower() == led_key.lower():
            return parse_values(node.get(k))
    for key in ["channels", "Channels", "data", "Data"]:
        if isinstance(node.get(key), dict):
            m = node[key]
            if led_key in m:
                return parse_values(m[led_key])
            for mk in m.keys():
                if isinstance(mk, str) and mk.lower() == led_key.lower():
                    return parse_values(m[mk])
    return []

def extract_bg_sample(json_obj, led_key="SC_Green"):
    """
    Find Background and Sample arrays for the chosen LED inside a single run JSON.
    Looks for entries where scanType ~ 'Background' or 'Sample' (case-insensitive),
    either in node['parameters']['scanType'] or node['scanType'].
    """
    scans = _find_scans(json_obj) or []
    bg_list, sm_list = [], []
    for node in scans:
        if not isinstance(node, dict):
            continue
        params = node.get("parameters") or {}
        stype = params.get("scanType") or node.get("scanType") or ""
        ch_vals = _get_channel_array(node, led_key)
        if not ch_vals:
            continue
        s_low = str(stype).lower()
        if s_low.startswith("back"):
            bg_list = ch_vals
        elif s_low.startswith("sam"):
            sm_list = ch_vals
    return bg_list, sm_list

def extract_sample_node(json_obj):
    """Return the first node in scans that is a Sample."""
    scans = _find_scans(json_obj) or []
    for node in scans:
        if not isinstance(node, dict):
            continue
        params = node.get("parameters") or {}
        stype = params.get("scanType") or node.get("scanType") or ""
        if str(stype).lower().startswith("sam"):
            return node
    return None

def get_loc_doses_from_sample(json_obj):
    """
    From the Sample scan node, extract LOC dosing volumes in µL.
    Accepts keys like 'LOC1', 'LOC2', ... with numeric values.
    Returns dict: {'LOC1': 200, 'LOC3': 2000, ...} for nonzero entries.
    """
    node = extract_sample_node(json_obj)
    doses = {}
    if node is None:
        return doses
    # Search in node and nested 'parameters' for LOC fields
    candidates = [node, node.get("parameters", {})]
    for src in candidates:
        if isinstance(src, dict):
            for k, v in src.items():
                if isinstance(k, str) and k.upper().startswith("LOC"):
                    try:
                        val = float(v)
                        if abs(val) > 0:
                            doses[k] = val
                    except Exception:
                        continue
    return doses

def compute_spike_concentration_mgL(stock_mgL, spike_uL, base_sample_mL=40.0, extra_reagent_mL=0.0):
    """
    Compute final concentration after spiking:
    C_final = C_stock * V_spike / V_total
    where V_spike is in mL (spike_uL/1000), and V_total = base_sample_mL + extra_reagent_mL.
    Caller can add total LOC volume to extra_reagent_mL if desired.
    """
    V_spike_mL = spike_uL / 1000.0
    V_total_mL = base_sample_mL + extra_reagent_mL
    if V_total_mL <= 0:
        return float("nan")
    return stock_mgL * (V_spike_mL / V_total_mL)

def compute_absorbance_from_json_bytes(file_bytes, led_key="SC_Green"):
    """
    Compute absorbance A from a single device JSON file:
    A = log10(mean(Background) / mean(Sample))
    using robust means (MAD filter).
    """
    try:
        obj = json.loads(file_bytes.decode("utf-8"))
    except Exception:
        obj = json.loads(file_bytes)
    bg_vals, sm_vals = extract_bg_sample(obj, led_key=led_key)
    if not bg_vals or not sm_vals:
        raise ValueError("Could not find both Background and Sample arrays for the selected LED in the JSON.")
    bg_avg, bg_kept = robust_mean(bg_vals)
    sm_avg, sm_kept = robust_mean(sm_vals)
    if bg_avg <= 0 or sm_avg <= 0:
        raise ValueError("Non-positive intensity after averaging.")
    A = math.log10(bg_avg / sm_avg)
    diag = {
        "bg_avg": bg_avg, "sm_avg": sm_avg,
        "bg_kept_n": len(bg_kept), "sm_kept_n": len(sm_kept),
        "bg_raw_n": len(bg_vals), "sm_raw_n": len(sm_vals),
    }
    return A, diag, obj  # return parsed obj for LOC use

def fit_linear(xs, ys, weights=None):
    """Weighted (or unweighted) linear regression. Returns (slope, intercept, R2)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if weights is None:
        w = np.ones_like(xs, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != xs.shape:
            w = np.ones_like(xs, dtype=float)
    W = np.sum(w)
    xw = np.sum(w * xs) / W
    yw = np.sum(w * ys) / W
    num = np.sum(w * (xs - xw) * (ys - yw))
    den = np.sum(w * (xs - xw) ** 2)
    if den == 0:
        raise ValueError("Zero variance in x.")
    m = num / den
    b = yw - m * xw
    ss_tot = np.sum(w * (ys - yw) ** 2)
    ss_res = np.sum(w * (ys - (m * xs + b)) ** 2)
    R2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(m), float(b), float(R2)

def lod_loq(blank_As, slope):
    """Compute LoD and LoQ from blank SD and slope."""
    if slope == 0 or not blank_As:
        return float("nan"), float("nan"), 0.0
    sd = float(np.std(blank_As, ddof=0))
    return 3.3 * sd / abs(slope), 10.0 * sd / abs(slope), sd

def predict_conc(A, slope, intercept):
    return (A - intercept) / slope

def make_plot(xs, ys, m, b, title, xlabel="Concentration (mg/L)", ylabel="Absorbance (A)"):
    """Create a simple scatter + fitted line plot, return PNG and PDF bytes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, ys)
    x_min, x_max = min(xs), max(xs)
    span = x_max - x_min
    grid_x = np.linspace(x_min - 0.05*span, x_max + 0.05*span if span > 0 else x_max + 1, 100)
    ax.plot(grid_x, m*grid_x + b)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=200, bbox_inches="tight")
    png_buf.seek(0)

    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)

    plt.close(fig)
    return png_buf.getvalue(), pdf_buf.getvalue()

# =========================
# LOC Profile (persistent mapping)
# =========================

DEFAULT_PROFILE = {
    "defaults": {
        "base_sample_mL": 40.0,
        "extra_constant_mL": 0.0,
        "include_all_loc_volumes": True
    },
    # "locs": {"LOC1": {"role":"standard","stock_mgL":1000.0,"notes":"Fe2 standard"}, ...}
    "locs": {}
}

def get_profile():
    if "loc_profile" not in st.session_state:
        st.session_state["loc_profile"] = json.loads(json.dumps(DEFAULT_PROFILE))
    return st.session_state["loc_profile"]

def set_profile(p):
    if "defaults" not in p:
        p["defaults"] = DEFAULT_PROFILE["defaults"].copy()
    if "locs" not in p:
        p["locs"] = {}
    st.session_state["loc_profile"] = p

def profile_pick_standard(detected_locs: dict):
    """
    Given detected LOC doses from a file, use the profile to decide which LOC is standard.
    Returns (std_loc, stock_mgL) or (None, None) if not determinable.
    """
    prof = get_profile()
    loc_map = prof.get("locs", {})
    candidates = [k for k in detected_locs.keys() if loc_map.get(k, {}).get("role") == "standard"]
    if not candidates:
        return None, None
    candidates.sort(key=lambda k: float(detected_locs.get(k, 0.0)), reverse=True)
    chosen = candidates[0]
    stock = float(loc_map.get(chosen, {}).get("stock_mgL", 0.0) or 0.0)
    if stock <= 0:
        return chosen, None
    return chosen, stock

# =========================
# UI
# =========================

st.title("Fe Phenanthroline — Method Development & Analysis")

with st.sidebar:
    st.header("Configuration")
    led_key = st.selectbox("LED channel", ["SC_Green","SC_Blue2","SC_Orange","SC_Red"], index=0)
    weighting_scheme = st.selectbox(
        "Weighting scheme",
        ["None (OLS)", "1/max(C,1)", "Variance-weighted (1/SD^2)"],
        index=2
    )
    use_rep_means = st.checkbox("Fit using replicate means", value=True)
    expected_reps = st.number_input("Expected replicates per level", min_value=1, max_value=10, value=2, step=1)
    st.caption("Replicates are used to average A at each level and compute SD/RSD.")

    with st.expander("LOC Profile (persistent)", expanded=False):
        prof = get_profile()

        # Defaults
        st.markdown("**Defaults**")
        prof["defaults"]["base_sample_mL"] = st.number_input(
            "Base sample volume (mL) — default", min_value=1.0, max_value=500.0,
            value=float(prof["defaults"].get("base_sample_mL", 40.0)), step=0.5
        )
        prof["defaults"]["extra_constant_mL"] = st.number_input(
            "Extra constant reagent volume per run (mL) — default", min_value=0.0, max_value=50.0,
            value=float(prof["defaults"].get("extra_constant_mL", 0.0)), step=0.1
        )
        prof["defaults"]["include_all_loc_volumes"] = st.checkbox(
            "Include ALL LOC volumes in final volume by default",
            value=bool(prof["defaults"].get("include_all_loc_volumes", True))
        )

        st.markdown("---")
        st.markdown("**LOC mappings**")
        for i in range(1, 17):
            key = f"LOC{i}"
            row = prof["locs"].get(key, {})
            cols = st.columns([1, 1, 1, 2])
            with cols[0]:
                options = ["", "standard", "reducer", "buffer", "other"]
                idx = options.index(row.get("role","")) if row.get("role","") in options else 0
                role = st.selectbox(f"{key} role", options, index=idx, key=f"role_{key}")
            with cols[1]:
                stock = st.number_input(f"{key} stock (mg/L)", min_value=0.0, max_value=1_000_000.0,
                                        value=float(row.get("stock_mgL", 0.0) or 0.0), step=10.0, key=f"stock_{key}")
            with cols[2]:
                note = st.text_input(f"{key} notes", value=row.get("notes",""), key=f"note_{key}")
            if role or stock > 0 or note:
                prof["locs"][key] = {"role": role or "", "stock_mgL": stock, "notes": note}
            else:
                if key in prof["locs"]:
                    del prof["locs"][key]

        st.markdown("---")
        colp1, colp2 = st.columns(2)
        with colp1:
            st.download_button("Download profile JSON", data=json.dumps(prof, indent=2),
                               file_name="loc_profile.json", mime="application/json")
        with colp2:
            up = st.file_uploader("Load profile JSON", type=["json"], key="loc_profile_upload")
            if up:
                try:
                    p = json.loads(up.getvalue().decode("utf-8"))
                    set_profile(p)
                    st.success("Profile loaded.")
                except Exception as e:
                    st.error(f"Failed to load profile: {e}")

tabs = st.tabs(["Calibration Builder", "Unknown Prediction", "DOE Plan", "JSON Explorer", "About"])

# ---------- Calibration Builder ----------
with tabs[0]:
    st.subheader("Upload calibration runs")
    st.write("Upload JSON files for **Fe²⁺** (no reducer) and **Total-Fe** (with reducer). Include several **0 mg/L blanks**.")
    uploaded_files = st.file_uploader("Drop multiple JSON files", type=["json"], accept_multiple_files=True)

    # Per-file concentration & channel prompt (with LOC-based inference)
    file_rows = []
    if uploaded_files:
        st.markdown("### Assign channel & concentration to each file")
        for f in uploaded_files:
            with st.expander(f"File: {f.name}", expanded=True):
                # Parse JSON
                try:
                    obj = json.loads(f.getvalue().decode("utf-8"))
                except Exception:
                    obj = json.loads(f.getvalue())

                # Extract LOC doses and try profile auto-pick
                loc_doses = get_loc_doses_from_sample(obj)
                prof = get_profile()
                auto_std_loc, auto_stock = profile_pick_standard(loc_doses) if loc_doses else (None, None)

                if loc_doses:
                    st.write("Detected LOC doses (µL):", loc_doses)
                    keys = list(loc_doses.keys())
                    options = ["(no standard)"] + keys
                    default_idx = 0
                    if auto_std_loc in keys:
                        default_idx = 1 + keys.index(auto_std_loc)
                        st.success(f"Profile suggests standard at **{auto_std_loc}**")
                    std_loc = st.selectbox(
                        f"Which LOC is the STANDARD in {f.name}?",
                        options, index=default_idx, key=f"stdloc_{f.name}"
                    )

                    # Volume defaults from profile
                    base_vol = st.number_input(
                        "Base sample volume (mL)",
                        min_value=1.0, max_value=200.0,
                        value=float(prof['defaults'].get('base_sample_mL', 40.0)),
                        step=0.5, key=f"base_{f.name}"
                    )
                    extra_const = st.number_input(
                        "Extra constant reagent volume per run (mL)",
                        min_value=0.0, max_value=20.0,
                        value=float(prof['defaults'].get('extra_constant_mL', 0.0)),
                        step=0.1, key=f"extra_{f.name}"
                    )
                    include_all_locs = st.checkbox(
                        "Include ALL LOC volumes in final volume (recommended)",
                        value=bool(prof['defaults'].get('include_all_loc_volumes', True)),
                        key=f"inclall_{f.name}"
                    )
                    total_loc_mL = sum(loc_doses.values())/1000.0 if include_all_locs else 0.0

                    if std_loc == "(no standard)":
                        conc_calc = 0.0
                        st.info("No standard spike selected → assigned **0.0000 mg/L** (blank).")
                    else:
                        # Stock concentration: prefill from profile if available
                        default_stock = None
                        if auto_std_loc and auto_std_loc == std_loc and auto_stock:
                            default_stock = float(auto_stock)
                        elif prof['locs'].get(std_loc, {}).get('stock_mgL', 0.0):
                            default_stock = float(prof['locs'][std_loc]['stock_mgL'])
                        stock_conc = st.number_input(
                            f"Stock concentration (mg/L) at {std_loc}",
                            min_value=0.0, max_value=1_000_000.0, value=(default_stock if default_stock is not None else 1000.0),
                            step=10.0, key=f"stock_{f.name}"
                        )
                        spike_uL = float(loc_doses.get(std_loc, 0.0))
                        conc_calc = compute_spike_concentration_mgL(
                            stock_conc, spike_uL,
                            base_sample_mL=base_vol,
                            extra_reagent_mL=extra_const + total_loc_mL
                        )
                        st.info(f"Calculated nominal concentration from {std_loc}: **{conc_calc:.4f} mg/L**")

                    use_auto = st.checkbox("Use this calculated concentration", value=True, key=f"useauto_{f.name}")
                else:
                    st.warning("No LOC doses detected in the Sample scan → assigned **0.0000 mg/L** (blank).")
                    conc_calc = 0.0
                    use_auto = True  # 0 by default

                ch = st.selectbox(f"Channel for {f.name}", ["Fe2", "TotalFe"], key=f"ch_{f.name}")
                manual_conc = st.number_input(
                    f"Manual concentration (mg/L) for {f.name} (overrides if provided)",
                    min_value=0.0, max_value=1000000.0, value=0.0, step=0.1, key=f"conc_{f.name}"
                )
                final_conc = manual_conc if manual_conc > 0 else (conc_calc if use_auto else 0.0)
                st.caption(f"Final concentration used for this file: {final_conc:.4f} mg/L")

                # Compute absorbance
                try:
                    A, diag, _ = compute_absorbance_from_json_bytes(f.getvalue(), led_key=led_key)
                    st.code(f"A = {A:.6f} | BG mean = {diag['bg_avg']:.2f}, Sample mean = {diag['sm_avg']:.2f} | N(BG)={diag['bg_kept_n']}/{diag['bg_raw_n']}, N(S)={diag['sm_kept_n']}/{diag['sm_raw_n']}")
                except Exception as e:
                    st.error(f"Absorbance calc failed: {e}")
                    A = None

                file_rows.append({"file_name": f.name, "channel": ch, "concentration_mgL": final_conc, "A": A})

        if st.button("Add all to calibration table"):
            cal_results = pd.DataFrame(file_rows)
            st.session_state["cal_results"] = cal_results

    # Show absorbances and summaries
    if "cal_results" in st.session_state:
        cal_df = st.session_state["cal_results"]
        st.markdown("### Calibration table")
        st.dataframe(cal_df)

        # Replicate aggregation
        def aggregate(df_in, channel_name):
            sub = df_in[df_in["channel"].str.lower() == channel_name.lower()].copy()
            sub["concentration_mgL"] = pd.to_numeric(sub["concentration_mgL"], errors="coerce")
            sub = sub.dropna(subset=["concentration_mgL", "A"])
            if sub.empty:
                return None, None
            grp = sub.groupby("concentration_mgL", as_index=False).agg(
                n=("A", "count"),
                A_mean=("A", "mean"),
                A_sd=("A", "std")
            )
            grp["A_sd"].fillna(0.0, inplace=True)
            grp["A_rsd_%"] = np.where(grp["A_mean"] != 0, grp["A_sd"] / grp["A_mean"] * 100.0, np.nan)
            grp["meets_n"] = grp["n"] >= expected_reps
            grp["flag"] = np.where(grp["meets_n"], "", f"Need ≥{expected_reps}")
            return sub, grp

        st.markdown("### Replicate summary by level")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Fe²⁺**")
            subA, grpA = aggregate(cal_df, "Fe2")
            if grpA is not None:
                st.dataframe(grpA)
        with c2:
            st.markdown("**Total-Fe**")
            subB, grpB = aggregate(cal_df, "TotalFe")
            if grpB is not None:
                st.dataframe(grpB)

        # Fitting helper
        def fit_channel(sub_df, grp_df, title):
            """Fit channel with sanity checks. Returns fit results and plot bytes or None."""
            if sub_df is None and grp_df is None:
                return None, None, None

            # Build x/y from replicate means or raw, plus blanks for LoD/LoQ
            use_means = use_rep_means and (grp_df is not None) and (not grp_df.empty)
            if use_means:
                xs = grp_df["concentration_mgL"].astype(float).values.tolist()
                ys = grp_df["A_mean"].astype(float).values.tolist()
                sds = grp_df["A_sd"].astype(float).values.tolist()
                blanks = grp_df[grp_df["concentration_mgL"] == 0]["A_mean"].astype(float).values.tolist()
            else:
                if sub_df is None or sub_df.empty:
                    return None, None, None
                xs = sub_df["concentration_mgL"].astype(float).values.tolist()
                ys = sub_df["A"].astype(float).values.tolist()
                sds = None  # not available at raw level
                blanks = sub_df[sub_df["concentration_mgL"] == 0]["A"].astype(float).values.tolist()

            # Drop NaNs/Infs
            xy = [(x, y, sds[i] if sds is not None and i < len(sds) else None) for i, (x, y) in enumerate(zip(xs, ys)) if np.isfinite(x) and np.isfinite(y)]
            if len(xy) < 2:
                st.warning(f"{title}: Need at least 2 valid data points to fit a line.")
                return None, None, None

            xs = [x for x, _, _ in xy]
            ys = [y for _, y, _ in xy]
            sds = [sd for _, _, sd in xy] if use_means else None

            # Require at least 2 unique levels
            if len(set(round(x, 6) for x in xs)) < 2:
                st.warning(f"{title}: All concentrations are the same. Add at least one more level.")
                return None, None, None

            # Build weights
            weights = None
            if weighting_scheme == "1/max(C,1)":
                weights = [1.0 / max(x, 1.0) for x in xs]
            elif weighting_scheme == "Variance-weighted (1/SD^2)" and use_means:
                eps = 1e-6
                nz = [sd for sd in sds if sd and sd > 0]
                base = np.median(nz) if nz else 0.01
                denom = [(sd if (sd and sd > 0) else base) ** 2 + eps for sd in sds]
                weights = [1.0 / d for d in denom]
            # else: OLS

            try:
                m, b, R2 = fit_linear(xs, ys, weights=weights)
            except Exception as e:
                st.error(f"{title}: Could not fit model: {e}")
                return None, None, None

            lod, loq, sd_blank = lod_loq(blanks, m)

            # Build plot
            try:
                png_bytes, pdf_bytes = make_plot(xs, ys, m, b, title=title)
            except Exception:
                png_bytes, pdf_bytes = None, None

            res = {
                "m": m, "b": b, "R2": R2,
                "LoD": lod, "LoQ": loq,
                "blank_sd_A": sd_blank,
                "n_points": len(xs),
                "levels": sorted(set(xs)),
                "weighting": weighting_scheme,
                "used_replicate_means": use_means,
            }
            return res, png_bytes, pdf_bytes

        st.markdown("### Model Fits")
        resA, pngA, pdfA = fit_channel(subA, grpA, "Fe²⁺ calibration")
        resB, pngB, pdfB = fit_channel(subB, grpB, "Total-Fe calibration")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Fe²⁺ model**")
            if resA:
                st.json(resA, expanded=False)
                if pngA:
                    st.download_button("Download Fe²⁺ plot (PNG)", data=pngA, file_name="Fe2_calibration.png", mime="image/png")
                    st.download_button("Download Fe²⁺ plot (PDF)", data=pdfA, file_name="Fe2_calibration.pdf", mime="application/pdf")
            else:
                st.info("Add Fe²⁺ rows with valid A and concentrations (≥2 distinct levels).")
        with cc2:
            st.markdown("**Total-Fe model**")
            if resB:
                st.json(resB, expanded=False)
                if pngB:
                    st.download_button("Download Total-Fe plot (PNG)", data=pngB, file_name="TotalFe_calibration.png", mime="image/png")
                    st.download_button("Download Total-Fe plot (PDF)", data=pdfB, file_name="TotalFe_calibration.pdf", mime="application/pdf")
            else:
                st.info("Add Total-Fe rows with valid A and concentrations (≥2 distinct levels).")

        if resA and resB:
            model = {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "led": led_key,
                "slope": {"Fe2": resA["m"], "TotalFe": resB["m"]},
                "intercept": {"Fe2": resA["b"], "TotalFe": resB["b"]},
                "R2": {"Fe2": resA["R2"], "TotalFe": resB["R2"]},
                "LoD": {"Fe2": resA["LoD"], "TotalFe": resB["LoD"]},
                "LoQ": {"Fe2": resA["LoQ"], "TotalFe": resB["LoQ"]},
                "blank_sd_A": {"Fe2": resA["blank_sd_A"], "TotalFe": resB["blank_sd_A"]},
                "range_mgL": [1.0, 25.0],
                "weighting_scheme": weighting_scheme,
                "use_replicate_means": resA["used_replicate_means"] and resB["used_replicate_means"],
                "expected_reps_per_level": expected_reps,
                "notes": "A = log10(mean(BG)/mean(Sample)); robust mean via MAD; replicate-aware; supports LOC-based concentration inference and variance-weighted regression."
            }
            st.markdown("### Download model JSON")
            st.download_button(
                "Download fe_model.json",
                data=json.dumps(model, indent=2),
                file_name="fe_model.json",
                mime="application/json",
            )

# ---------- Unknown Prediction ----------
with tabs[1]:
    st.subheader("Predict concentrations for unknown samples")
    st.write("Upload a **model JSON** and one or two run files. If you upload both Fe²⁺ and Total-Fe runs, the app will compute Fe³⁺ by difference.")
    model_file = st.file_uploader("Model JSON", type=["json"], key="model_upload")
    run_files = st.file_uploader("Unknown run JSON(s)", type=["json"], accept_multiple_files=True, key="unknown_runs")
    if model_file and run_files:
        try:
            model = json.loads(model_file.getvalue().decode("utf-8"))
        except Exception:
            model = json.loads(model_file.getvalue())
        led_m = model.get("led", "SC_Green")
        A_vals = {}
        details = {}
        for f in run_files:
            A, diag, _ = compute_absorbance_from_json_bytes(f.getvalue(), led_key=led_m)
            details[f.name] = {"A": A, **diag}
            guess = "Fe2"
            nm = f.name.lower()
            if "total" in nm or "tfe" in nm or "fe3" in nm:
                guess = "TotalFe"
            A_vals[guess] = A
        st.markdown("### Run details")
        st.json(details, expanded=False)
        out = {}
        if "Fe2" in A_vals:
            m = model["slope"]["Fe2"]; b = model["intercept"]["Fe2"]
            out["Fe2_mgL"] = predict_conc(A_vals["Fe2"], m, b)
        if "TotalFe" in A_vals:
            m = model["slope"]["TotalFe"]; b = model["intercept"]["TotalFe"]
            out["TotalFe_mgL"] = predict_conc(A_vals["TotalFe"], m, b)
        if "Fe2_mgL" in out and "TotalFe_mgL" in out:
            out["Fe3_mgL"] = out["TotalFe_mgL"] - out["Fe2_mgL"]
        st.markdown("### Results")
        st.json(out, expanded=False)

# ---------- DOE Plan ----------
with tabs[2]:
    st.subheader("Generate a randomized DOE plan")
    st.write("Choose levels (mg/L) and replicates; download a CSV plan with randomized order for Fe²⁺ and Total-Fe.")
    default_levels = [0, 1, 2, 5, 10, 15, 20, 25]
    levels_str = st.text_input("Levels (comma-separated mg/L)", ",".join(map(str, default_levels)))
    reps = st.number_input("Replicates per level", min_value=1, max_value=5, value=2, step=1)
    if st.button("Build plan"):
        try:
            levels = [float(x.strip()) for x in levels_str.split(",") if x.strip() != ""]
        except Exception:
            st.error("Could not parse levels. Using defaults.")
            levels = default_levels
        rows = []
        for ch in ["Fe2", "TotalFe"]:
            pts = []
            for _ in range(int(reps)):
                pts.extend(levels)
            import random
            random.shuffle(pts)
            for c in pts:
                rows.append({
                    "order": len(rows) + 1,
                    "channel": ch,
                    "target_mgL": c,
                    "json_path": "",
                    "notes": ""
                })
        df = pd.DataFrame(rows)
        st.dataframe(df)
        st.download_button("Download DOE CSV", data=df.to_csv(index=False), file_name="doe_plan.csv", mime="text/csv")

# ---------- JSON Explorer ----------
with tabs[3]:
    st.subheader("Inspect a single JSON")
    f = st.file_uploader("Upload a run JSON", type=["json"], key="explore_json")
    led_sel = st.selectbox("LED for this view", ["SC_Green","SC_Blue2","SC_Orange","SC_Red"], index=0, key="expl_led")
    if f:
        try:
            A, diag, obj = compute_absorbance_from_json_bytes(f.getvalue(), led_key=led_sel)
            st.success(f"Absorbance (A) at {led_sel}: {A:.6f}")
            st.json(diag, expanded=False)
            locs = get_loc_doses_from_sample(obj)
            if locs:
                st.write("Detected LOC doses (µL):", locs)
            st.table({k: [v] for k, v in diag.items()})
        except Exception as e:
            st.error(str(e))

# ---------- About ----------
with tabs[4]:
    st.markdown("""
**Fe Phenanthroline Calibration & Analysis**  
- Parses device JSON with *Background* and *Sample* in the **same file** (even if `scans` is nested under `payload`).  
- Uses **SC_Green** (or selectable) channel; averages 10 readings with MAD outlier filtering.  
- Absorbance: `A = log10(mean(BG) / mean(Sample))`.  
- Can **infer calibration concentrations from LOC doses** via \(M_1V_1=M_2V_2\) using a **LOC Profile** (persistent mapping of roles/stock) or manual entry.  
- Supports **replicate-aware** summaries, **variance-weighted (1/SD²)** or **1/max(C,1)** regression, and exports **PNG/PDF** calibration plots.  
- If **no standard spike** is detected or selected, the run is treated as a **0 mg/L blank** automatically.
""")
    st.caption("Tip: Include ≥4 blanks spread across the run to stabilize LoD/LoQ.")
