
import streamlit as st
import json
import math
import statistics as stats
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fe Phenanthroline Calibration & Analysis", layout="wide")

# ----------------- Utilities -----------------

def parse_values(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except ValueError:
            try:
                arr = json.loads(raw)
                return [float(x) for x in arr]
            except Exception:
                return []
    return []

def robust_mean(values):
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

def extract_bg_sample(json_obj, led_key="SC_Green"):
    scans = json_obj.get("scans", [])
    bg_list, sm_list = [], []
    for node in scans:
        stype = None
        if isinstance(node, dict):
            params = node.get("parameters") or {}
            stype = params.get("scanType") or node.get("scanType")
            ch_vals = node.get(led_key)
            if ch_vals is None:
                for k in node.keys():
                    if k.lower() == led_key.lower():
                        ch_vals = node[k]
                        break
            vals = parse_values(ch_vals)
            if not vals:
                continue
            if isinstance(stype, str) and stype.lower().startswith("back"):
                bg_list = vals
            elif isinstance(stype, str) and stype.lower().startswith("sam"):
                sm_list = vals
    return bg_list, sm_list

def compute_absorbance_from_json_bytes(file_bytes, led_key="SC_Green"):
    obj = json.loads(file_bytes.decode("utf-8"))
    bg_vals, sm_vals = extract_bg_sample(obj, led_key=led_key)
    if not bg_vals or not sm_vals:
        raise ValueError("Could not find both Background and Sample SC_Green arrays in the JSON.")
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
    return A, diag

def fit_linear(xs, ys, weights=None):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if weights is None:
        w = np.ones_like(xs)
    else:
        w = np.asarray(weights, dtype=float)
    W = np.sum(w)
    xw = np.sum(w*xs)/W
    yw = np.sum(w*ys)/W
    num = np.sum(w*(xs - xw)*(ys - yw))
    den = np.sum(w*(xs - xw)**2)
    if den == 0:
        raise ValueError("Zero variance in x.")
    m = num/den
    b = yw - m*xw
    ss_tot = np.sum(w*(ys - yw)**2)
    ss_res = np.sum(w*(ys - (m*xs + b))**2)
    R2 = 1.0 - (ss_res/ss_tot if ss_tot > 0 else 0.0)
    return float(m), float(b), float(R2)

def lod_loq(blank_As, slope):
    if slope == 0 or not blank_As:
        return float("nan"), float("nan"), 0.0
    sd = float(np.std(blank_As, ddof=0))
    return 3.3*sd/abs(slope), 10.0*sd/abs(slope), sd

def model_to_json(mA,bA,R2A, lodA, loqA, mB,bB,R2B, lodB, loqB, led="SC_Green"):
    return {
        "created_at": datetime.utcnow().isoformat()+"Z",
        "led": led,
        "slope": {"Fe2": mA, "TotalFe": mB},
        "intercept": {"Fe2": bA, "TotalFe": bB},
        "R2": {"Fe2": R2A, "TotalFe": R2B},
        "LoD": {"Fe2": lodA, "TotalFe": lodB},
        "LoQ": {"Fe2": loqA, "TotalFe": loqB},
        "range_mgL": [1.0, 25.0],
        "notes": "Absorbance A = log10(mean(BG)/mean(Sample)); robust mean with MAD; replicate-aware; weighted linear fit optional."
    }

def predict_conc(A, slope, intercept):
    return (A - intercept)/slope

# ----------------- UI -----------------

st.title("Fe Phenanthroline — Method Development & Analysis")

with st.sidebar:
    st.header("Configuration")
    led_key = st.selectbox("LED channel", ["SC_Green","SC_Blue2","SC_Orange","SC_Red"], index=0)
    use_weighting = st.checkbox("Weighted linear fit (1/max(C,1))", value=True)
    use_rep_means = st.checkbox("Fit using replicate means", value=True)
    expected_reps = st.number_input("Expected replicates per level", min_value=1, max_value=10, value=2, step=1)
    st.caption("Replicates are used to average A at each level and compute SD/RSD.")

tabs = st.tabs(["Calibration Builder","Unknown Prediction","DOE Plan","JSON Explorer","About"])

# ---------- Calibration Builder ----------
with tabs[0]:
    st.subheader("Upload calibration runs")
    st.write("Upload JSON files for **Fe²⁺** (no reducer) and **Total-Fe** (with reducer). Include several **0 mg/L blanks**.")
    uploaded_files = st.file_uploader("Drop multiple JSON files", type=["json"], accept_multiple_files=True)
    if "cal_rows" not in st.session_state:
        st.session_state.cal_rows = []

    new_rows = []
    for f in uploaded_files or []:
        name = f.name
        ch_guess = "Fe2" if "fe2" in name.lower() else ("TotalFe" if ("tfe" in name.lower() or "total" in name.lower() or "fe3" in name.lower()) else "Fe2")
        conc_guess = ""
        for tok in ["_0","_1","_2","_5","_10","_15","_20","_25","0mg","1mg","2mg","5mg","10mg","15mg","20mg","25mg"]:
            if tok in name.lower():
                try:
                    conc_guess = float(tok.replace("mg","").replace("_",""))
                except:
                    pass
        new_rows.append({"file_name": name, "channel": ch_guess, "concentration_mgL": conc_guess, "attach": f})
    if new_rows:
        st.session_state.cal_rows.extend(new_rows)

    if st.session_state.cal_rows:
        df = pd.DataFrame([{k: v for k, v in row.items() if k != "attach"} for row in st.session_state.cal_rows])
        st.write("Edit the channel and concentration values as needed:")
        edited = st.data_editor(df, num_rows="dynamic", key="cal_edit")
        for i, row in enumerate(st.session_state.cal_rows):
            for k in ["channel","concentration_mgL","file_name"]:
                row[k] = edited.iloc[i][k]

        if st.button("Compute absorbances for all rows"):
            results = []
            for row in st.session_state.cal_rows:
                f = next((x["attach"] for x in st.session_state.cal_rows if x["file_name"] == row["file_name"]), None)
                if f is None:
                    continue
                try:
                    A, diag = compute_absorbance_from_json_bytes(f.getvalue(), led_key=led_key)
                    results.append({**row, "A": A, **diag})
                except Exception as e:
                    results.append({**row, "A": None, "error": str(e)})
            st.session_state.cal_results = pd.DataFrame(results)

        if "cal_results" in st.session_state:
            st.markdown("### Absorbances")
            st.dataframe(st.session_state.cal_results.drop(columns=["attach"], errors="ignore"))

            # ---- Replicate aggregation ----
            def aggregate(df_in, channel_name):
                sub = df_in[df_in["channel"].str.lower() == channel_name.lower()].copy()
                sub["concentration_mgL"] = pd.to_numeric(sub["concentration_mgL"], errors="coerce")
                sub = sub.dropna(subset=["concentration_mgL","A"])
                if sub.empty:
                    return None, None, None
                grp = sub.groupby("concentration_mgL", as_index=False).agg(
                    n=("A","count"),
                    A_mean=("A","mean"),
                    A_sd=("A","std")
                )
                grp["A_sd"].fillna(0.0, inplace=True)
                grp["A_rsd_%"] = np.where(grp["A_mean"]!=0, grp["A_sd"]/grp["A_mean"]*100.0, np.nan)
                grp["meets_n"] = grp["n"] >= expected_reps
                grp["flag"] = np.where(grp["meets_n"], "", f"Need ≥{expected_reps}")
                return sub, grp, grp["A_mean"].values.tolist()

            st.markdown("### Replicate summary by level")
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Fe²⁺**")
                subA, grpA, yA = aggregate(st.session_state.cal_results, "Fe2")
                if grpA is not None:
                    st.dataframe(grpA)
            with cols[1]:
                st.markdown("**Total-Fe**")
                subB, grpB, yB = aggregate(st.session_state.cal_results, "TotalFe")
                if grpB is not None:
                    st.dataframe(grpB)

            # ---- Fit models (using replicate means if selected) ----
            def fit_channel(sub_df, grp_df):
                if sub_df is None:
                    return None
                if use_rep_means and grp_df is not None:
                    xs = grp_df["concentration_mgL"].values.tolist()
                    ys = grp_df["A_mean"].values.tolist()
                    blanks = grp_df[grp_df["concentration_mgL"] == 0]["A_mean"].values.tolist()
                else:
                    xs = sub_df["concentration_mgL"].values.tolist()
                    ys = sub_df["A"].values.tolist()
                    blanks = sub_df[sub_df["concentration_mgL"] == 0]["A"].values.tolist()
                weights = None
                if use_weighting:
                    weights = [1.0/max(x,1.0) for x in xs]
                m,b,R2 = fit_linear(xs, ys, weights=weights)
                lod, loq, sd_blank = lod_loq(blanks, m)
                return {"m": m, "b": b, "R2": R2, "LoD": lod, "LoQ": loq, "blank_sd_A": sd_blank,
                        "n_points": len(xs), "levels": sorted(set(xs))}

            st.markdown("### Model Fits")
            resA = fit_channel(subA, grpA)
            resB = fit_channel(subB, grpB)
            cols2 = st.columns(2)
            with cols2[0]:
                st.markdown("**Fe²⁺ model**")
                if resA: st.json(resA, expanded=False)
                else: st.info("Add Fe2 rows with valid A and concentrations.")
            with cols2[1]:
                st.markdown("**Total-Fe model**")
                if resB: st.json(resB, expanded=False)
                else: st.info("Add TotalFe rows with valid A and concentrations.")

            if resA and resB:
                model = {
                    "created_at": datetime.utcnow().isoformat()+"Z",
                    "led": led_key,
                    "slope": {"Fe2": resA["m"], "TotalFe": resB["m"]},
                    "intercept": {"Fe2": resA["b"], "TotalFe": resB["b"]},
                    "R2": {"Fe2": resA["R2"], "TotalFe": resB["R2"]},
                    "LoD": {"Fe2": resA["LoD"], "TotalFe": resB["LoD"]},
                    "LoQ": {"Fe2": resA["LoQ"], "TotalFe": resB["LoQ"]},
                    "blank_sd_A": {"Fe2": resA["blank_sd_A"], "TotalFe": resB["blank_sd_A"]},
                    "range_mgL": [1.0, 25.0],
                    "use_weighting": use_weighting,
                    "use_replicate_means": use_rep_means,
                    "expected_reps_per_level": expected_reps,
                }
                st.markdown("### Download model JSON")
                st.download_button("Download fe_model.json", data=json.dumps(model, indent=2),
                                   file_name="fe_model.json", mime="application/json")

# ---------- Unknown Prediction ----------
with tabs[1]:
    st.subheader("Predict concentrations for unknown samples")
    st.write("Upload a **model JSON** and one or two run files. If you upload both Fe²⁺ and Total-Fe runs, the app will compute Fe³⁺ by difference.")
    model_file = st.file_uploader("Model JSON", type=["json"], key="model_upload")
    run_files = st.file_uploader("Unknown run JSON(s)", type=["json"], accept_multiple_files=True, key="unknown_runs")
    if model_file and run_files:
        model = json.loads(model_file.getvalue().decode("utf-8"))
        led_m = model.get("led","SC_Green")
        A_vals = {}
        details = {}
        for f in run_files:
            A, diag = compute_absorbance_from_json_bytes(f.getvalue(), led_key=led_m)
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
            out["Fe2_mgL"] = (A_vals["Fe2"] - b)/m
        if "TotalFe" in A_vals:
            m = model["slope"]["TotalFe"]; b = model["intercept"]["TotalFe"]
            out["TotalFe_mgL"] = (A_vals["TotalFe"] - b)/m
        if "Fe2_mgL" in out and "TotalFe_mgL" in out:
            out["Fe3_mgL"] = out["TotalFe_mgL"] - out["Fe2_mgL"]
        st.markdown("### Results")
        st.json(out, expanded=False)

# ---------- DOE Plan ----------
with tabs[2]:
    st.subheader("Generate a randomized DOE plan")
    st.write("Choose levels (mg/L) and replicates; download a CSV plan with randomized order for Fe²⁺ and Total-Fe.")
    default_levels = [0,1,2,5,10,15,20,25]
    levels_str = st.text_input("Levels (comma-separated mg/L)", ",".join(map(str, default_levels)))
    reps = st.number_input("Replicates per level", min_value=1, max_value=5, value=2, step=1)
    if st.button("Build plan"):
        try:
            levels = [float(x.strip()) for x in levels_str.split(",") if x.strip()!=""]
        except Exception:
            st.error("Could not parse levels.")
            levels = default_levels
        rows = []
        for ch in ["Fe2","TotalFe"]:
            pts = []
            for _ in range(int(reps)):
                pts.extend(levels)
            import random
            random.shuffle(pts)
            for i, c in enumerate(pts, start=1):
                rows.append({"order": len(rows)+1, "channel": ch, "target_mgL": c, "json_path": "", "notes": ""})
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
            A, diag = compute_absorbance_from_json_bytes(f.getvalue(), led_key=led_sel)
            st.success(f"Absorbance (A) at {led_sel}: {A:.6f}")
            st.json(diag, expanded=False)
            st.table({k:[v] for k,v in diag.items()})
        except Exception as e:
            st.error(str(e))

# ---------- About ----------
with tabs[4]:
    st.markdown("""
**Fe Phenanthroline Calibration & Analysis**  
- Parses device JSON with *Background* and *Sample* scans.  
- Uses **SC_Green** (or selectable) channel; averages 10 readings with MAD outlier filtering.  
- Absorbance: `A = log10(mean(BG) / mean(Sample))`.  
- **Replicate-aware calibration**: specify expected replicates; app summarizes mean/SD/RSD per level and fits using replicate means (optional).  
- Computes **LoD / LoQ** from blank SD; predicts unknowns; reports **Fe³⁺ by difference**.  
""")
    st.caption("Tip: Include ≥4 blanks spread across the run to stabilize LoD/LoQ.")
