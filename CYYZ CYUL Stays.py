# overnights_cyyz.py
import io
from datetime import datetime, timedelta, time
import pandas as pd
import pytz
import streamlit as st

# ===============================
# Page config & title
# ===============================
st.set_page_config(page_title="CYYZ Overnights Calculator", layout="wide")
st.title("CYYZ Overnights Calculator")
st.caption("Upload FL3XX arrivals/departures CSVs for a month and compute overnight counts by day (two metrics).")

# ===============================
# Helpers
# ===============================
@st.cache_data(show_spinner=False)
def flexible_read_csv(file) -> pd.DataFrame:
    """Try common delimiters; fall back to default."""
    if file is None:
        return pd.DataFrame()
    content = file.read()
    bio = io.BytesIO(content)
    for sep in [",", ";", "\t"]:
        try:
            bio.seek(0)
            df = pd.read_csv(bio, sep=sep, engine="python")
            # If we guessed wrong (single col but separators inside), try next
            if df.shape[1] == 1 and any(s in str(df.iloc[0, 0]) for s in [",", ";", "\t"]):
                continue
            return df
        except Exception:
            continue
    bio.seek(0)
    return pd.read_csv(bio)

def pick_col(cols, preferred_list):
    """Return the first exact (case-insensitive) match; else first 'contains' match; else None."""
    low = {c.lower(): c for c in cols}
    for p in preferred_list:
        if p.lower() in low:
            return low[p.lower()]
    for p in preferred_list:
        for c in cols:
            if p.lower() in c.lower():
                return c
    return None

def parse_flexible_utc_to_local(series: pd.Series, local_tz: pytz.timezone, dayfirst: bool = True) -> pd.Series:
    """
    Robust parser for FL3XX-like strings, e.g. '01.08.2025 20:48z'
    - normalizes trailing 'z'/'Z' to '+00:00'
    - supports dot-separated day-first dates
    - converts to the provided local timezone
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r'(?i)z$', '+00:00', regex=True)  # normalize trailing Z
    dt_utc = pd.to_datetime(s, errors="coerce", utc=True, dayfirst=dayfirst, infer_datetime_format=True)
    return dt_utc.dt.tz_convert(local_tz)

def localize_naive(series: pd.Series, local_tz: pytz.timezone) -> pd.Series:
    """Treat timestamps as local; handle naive or already tz-aware values."""
    out = []
    for v in pd.to_datetime(series, errors="coerce", infer_datetime_format=True):
        if pd.isna(v):
            out.append(pd.NaT)
        else:
            out.append(local_tz.localize(v) if v.tzinfo is None else v.astimezone(local_tz))
    return pd.Series(out)

def merge_intervals(intervals):
    """Intervals: list[(start_dt, end_dt)] in same tz. Returns merged sorted list."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged

def overlap(a_start, a_end, b_start, b_end):
    s = max(a_start, b_start)
    e = min(a_end, b_end)
    return max(timedelta(0), e - s)

# ===============================
# Inputs
# ===============================
st.subheader("1) Upload CSVs")
col_u1, col_u2 = st.columns(2)
with col_u1:
    f_arrivals = st.file_uploader("Arrivals CSV (TO CYYZ …)", type=["csv"], key="arr")
with col_u2:
    f_departures = st.file_uploader("Departures CSV (FROM CYYZ …)", type=["csv"], key="dep")

arr_raw = flexible_read_csv(f_arrivals) if f_arrivals else pd.DataFrame()
dep_raw = flexible_read_csv(f_departures) if f_departures else pd.DataFrame()

if not arr_raw.empty:
    with st.expander("Preview: Arrivals (first 20 rows)"):
        st.dataframe(arr_raw.head(20), use_container_width=True)
if not dep_raw.empty:
    with st.expander("Preview: Departures (first 20 rows)"):
        st.dataframe(dep_raw.head(20), use_container_width=True)

st.subheader("2) Settings")
col_a, col_b, col_c = st.columns([1.2, 1, 1])
with col_a:
    airport_input = st.text_input(
        "Airports (ICAO — comma separated)", value="CYYZ"
    )
    airports = []
    for token in airport_input.split(","):
        code = token.strip().upper()
        if code and code not in airports:
            airports.append(code)
with col_b:
    month_int = st.number_input("Month (1–12)", min_value=1, max_value=12, value=8, step=1)
with col_c:
    year_int = st.number_input("Year", min_value=2000, max_value=2100, value=datetime.now().year, step=1)

col_tz1, col_tz2 = st.columns(2)
with col_tz1:
    ts_source_tz = st.selectbox(
        "Timestamps in files are:",
        ["UTC (recommended)", "Local (choose zone below)"],
        index=0
    )
with col_tz2:
    local_tz_name = st.selectbox(
        "Local timezone (for calculations & display)",
        ["America/Toronto", "America/Edmonton", "UTC"] + sorted(pytz.all_timezones),
        index=0
    )
LOCAL_TZ = pytz.timezone(local_tz_name)

# New: day-first parsing toggle (for '01.08.2025' = 1 Aug 2025)
dayfirst_ui = st.checkbox("Parse dates as day-first (e.g., 01.08.2025 = 1 Aug 2025)", value=True)

st.caption("All calculations are performed in the selected **Local timezone** (default America/Toronto).")

# Column mapping UI
if not arr_raw.empty and not dep_raw.empty:
    st.subheader("3) Column Mapping")
    arr_cols = list(arr_raw.columns)
    dep_cols = list(dep_raw.columns)

    # Sensible defaults (prefer Actual)
    default_arr_tail = pick_col(arr_cols, ["Aircraft", "Tail", "Registration", "A/C"])
    default_dep_tail = pick_col(dep_cols, ["Aircraft", "Tail", "Registration", "A/C"])
    default_to = pick_col(arr_cols, ["To (ICAO)", "To", "Destination"])
    default_from = pick_col(dep_cols, ["From (ICAO)", "From", "Origin"])
    default_arr_time = pick_col(arr_cols, ["On-Block (Act)", "On-Block (Actual)", "On-Block", "ATA", "Arrival (UTC)"])
    default_dep_time = pick_col(dep_cols, ["Off-Block (Act)", "Off-Block (Actual)", "Off-Block", "ATD", "Departure (UTC)"])

    c1, c2 = st.columns(2)
    with c1:
        tail_arr = st.selectbox("Arrivals: Tail/Registration", arr_cols, index=arr_cols.index(default_arr_tail) if default_arr_tail in arr_cols else 0)
        arr_to_col = st.selectbox("Arrivals: To (ICAO)", arr_cols, index=arr_cols.index(default_to) if default_to in arr_cols else 0)
        arr_time_col = st.selectbox("Arrivals: Arrival time", arr_cols, index=arr_cols.index(default_arr_time) if default_arr_time in arr_cols else 0)
    with c2:
        tail_dep = st.selectbox("Departures: Tail/Registration", dep_cols, index=dep_cols.index(default_dep_tail) if default_dep_tail in dep_cols else 0)
        dep_from_col = st.selectbox("Departures: From (ICAO)", dep_cols, index=dep_cols.index(default_from) if default_from in dep_cols else 0)
        dep_time_col = st.selectbox("Departures: Departure time", dep_cols, index=dep_cols.index(default_dep_time) if default_dep_time in dep_cols else 0)

    st.subheader("4) Overnight Definitions")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        check_hour = st.time_input("Metric A — On ground at (local time)", value=time(3, 0))
    with col_m2:
        night_start = st.time_input("Metric B — Night window start (local)", value=time(20, 0))
        night_end = st.time_input("Metric B — Night window end (local, next day)", value=time(6, 0))
    threshold_hours = st.slider("Metric B — Minimum on-ground within night window (hours)", 0.0, 12.0, 5.0, 0.5)

    st.subheader("5) Pairing Options")
    assume_from_month_start = st.checkbox(
        "Assume a tail with departures but no earlier arrivals this month was already on-ground from month start",
        value=False,
        help="Enable if you want to count aircraft that only show up as departures (or whose first arrival is later in the month) as being present from the start of the month."
    )

    st.subheader("6) Run")
    if st.button("Compute Overnights"):
        if not airports:
            st.error("Enter at least one airport code (e.g., CYYZ, CYUL).")
            st.stop()

        # ===============================
        # Parse & normalize rows
        # ===============================
        arr = arr_raw.copy()
        dep = dep_raw.copy()

        arr["airport_dest"] = arr[arr_to_col].astype(str).str.strip().str.upper()
        dep["airport_origin"] = dep[dep_from_col].astype(str).str.strip().str.upper()

        # Parse timestamps from source tz to LOCAL_TZ
        if ts_source_tz.startswith("UTC"):
            arr["arr_dt"] = parse_flexible_utc_to_local(arr[arr_time_col], LOCAL_TZ, dayfirst=dayfirst_ui)
            dep["dep_dt"] = parse_flexible_utc_to_local(dep[dep_time_col], LOCAL_TZ, dayfirst=dayfirst_ui)
        else:
            arr["arr_dt"] = localize_naive(arr[arr_time_col], LOCAL_TZ)
            dep["dep_dt"] = localize_naive(dep[dep_time_col], LOCAL_TZ)

        # Sanity check BEFORE month filter
        all_parsed = pd.concat([arr["arr_dt"].dropna(), dep["dep_dt"].dropna()])
        if not all_parsed.empty:
            month_hits = ((all_parsed.dt.year == int(year_int)) & (all_parsed.dt.month == int(month_int))).mean()
            if month_hits < 0.2:
                st.warning(
                    "Heads up: Most parsed timestamps are NOT in the chosen month/year. "
                    "This usually means the date format (day-first) toggle is wrong or the files use a different month."
                )

        # Normalize tail
        arr["tail"] = arr[tail_arr].astype(str).str.strip().str.upper()
        dep["tail"] = dep[tail_dep].astype(str).str.strip().str.upper()

        # Drop missing datetimes
        arr = arr.dropna(subset=["arr_dt"])
        dep = dep.dropna(subset=["dep_dt"])

        # Month window
        start_local = LOCAL_TZ.localize(datetime(int(year_int), int(month_int), 1, 0, 0, 0))
        if int(month_int) == 12:
            end_month_start = LOCAL_TZ.localize(datetime(int(year_int) + 1, 1, 1, 0, 0, 0))
        else:
            end_month_start = LOCAL_TZ.localize(datetime(int(year_int), int(month_int) + 1, 1, 0, 0, 0))
        end_local = end_month_start - timedelta(seconds=1)

        # Keep only events inside month (helps pairing clarity)
        arr = arr[(arr["arr_dt"] >= start_local) & (arr["arr_dt"] <= end_local)].copy()
        dep = dep[(dep["dep_dt"] >= start_local) & (dep["dep_dt"] <= end_local)].copy()

        def compute_airport_metrics(airport_code: str):
            arr_filtered = arr[arr["airport_dest"] == airport_code].copy()
            dep_filtered = dep[dep["airport_origin"] == airport_code].copy()

            arr_filtered = arr_filtered.sort_values(["tail", "arr_dt"])
            dep_filtered = dep_filtered.sort_values(["tail", "dep_dt"])
            dep_map = {t: g["dep_dt"].tolist() for t, g in dep_filtered.groupby("tail")}

            intervals_by_tail = {}
            for tail, g in arr_filtered.groupby("tail"):
                dep_times = dep_map.get(tail, [])
                j = 0
                own_intervals = []
                for arr_t in g["arr_dt"]:
                    dep_t = None
                    while j < len(dep_times) and dep_times[j] <= arr_t:
                        j += 1
                    if j < len(dep_times) and dep_times[j] > arr_t:
                        dep_t = dep_times[j]
                        j += 1
                    else:
                        dep_t = end_local  # open through month end
                    own_intervals.append((arr_t, dep_t))
                intervals_by_tail[tail] = own_intervals

            if assume_from_month_start:
                first_arrivals = {tail: g["arr_dt"].min() for tail, g in arr_filtered.groupby("tail")}
                for tail, dep_times in dep_filtered.groupby("tail"):
                    first_dep = dep_times["dep_dt"].min()
                    first_arr = first_arrivals.get(tail, pd.NaT)
                    if pd.notna(first_dep) and first_dep > start_local:
                        if pd.isna(first_arr) or first_arr > first_dep:
                            intervals_by_tail.setdefault(tail, []).append((start_local, first_dep))

            for tail, ivls in list(intervals_by_tail.items()):
                clipped = []
                for s, e in ivls:
                    if e < start_local or s > end_local:
                        continue
                    s2 = max(s, start_local)
                    e2 = min(e, end_local)
                    if e2 > s2:
                        clipped.append((s2, e2))
                intervals_by_tail[tail] = merge_intervals(clipped)

            dates = pd.date_range(start_local.date(), end_local.date(), freq="D")
            rows_A = []
            for d in dates:
                check_dt = LOCAL_TZ.localize(datetime(d.year, d.month, d.day, check_hour.hour, check_hour.minute)) + timedelta(days=1)
                present = [t for t, ivls in intervals_by_tail.items() if any(s <= check_dt <= e for s, e in ivls)]
                rows_A.append({
                    "Date": pd.to_datetime(d),
                    "Overnights_A_check": len(present),
                    "Tails_A": ", ".join(sorted(present))
                })
            df_A = pd.DataFrame(rows_A)

            rows_B = []
            for d in dates:
                win_start = LOCAL_TZ.localize(datetime(d.year, d.month, d.day, night_start.hour, night_start.minute))
                if night_end <= night_start:
                    win_end = LOCAL_TZ.localize(datetime(d.year, d.month, d.day, night_end.hour, night_end.minute)) + timedelta(days=1)
                else:
                    win_end = LOCAL_TZ.localize(datetime(d.year, d.month, d.day, night_end.hour, night_end.minute))

                present_B = []
                for tail, ivls in intervals_by_tail.items():
                    total = timedelta(0)
                    for s, e in ivls:
                        total += overlap(s, e, win_start, win_end)
                        if total >= timedelta(hours=float(threshold_hours)):
                            present_B.append(tail)
                            break
                rows_B.append({
                    "Date": pd.to_datetime(d),
                    "Overnights_B_nightwindow": len(present_B),
                    "Tails_B": ", ".join(sorted(present_B))
                })
            df_B = pd.DataFrame(rows_B)

            combined = df_A.merge(df_B, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
            combined["Delta_B_minus_A"] = combined["Overnights_B_nightwindow"] - combined["Overnights_A_check"]

            diag_rows = []
            for tail, ivls in sorted(intervals_by_tail.items()):
                for s, e in ivls:
                    diag_rows.append({
                        "Tail": tail,
                        "OnGround_Start_Local": s,
                        "OnGround_End_Local": e,
                        "Hours": (e - s).total_seconds() / 3600.0
                    })
            diagnostics = pd.DataFrame(diag_rows).sort_values(["Tail", "OnGround_Start_Local"]).reset_index(drop=True)

            return combined, diagnostics

        metrics = {apt: compute_airport_metrics(apt) for apt in airports}

        st.success(f"Computed for {', '.join(airports)}!")

        st.subheader("Results")
        tabs = st.tabs([f"{apt}" for apt in airports])
        for tab, apt in zip(tabs, airports):
            combined, diagnostics = metrics[apt]
            with tab:
                st.markdown(f"### {apt}")
                st.dataframe(combined, use_container_width=True)

                st.subheader("Diagnostics (on-ground intervals per tail)")
                st.caption("Use this to spot-check pairings and durations. Filter by Tail with the column tools.")
                st.dataframe(diagnostics, use_container_width=True, height=400)

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        "Download results (CSV)",
                        data=combined.to_csv(index=False).encode("utf-8"),
                        file_name=f"{apt}_overnights_{int(year_int)}-{int(month_int):02d}_metrics.csv",
                        mime="text/csv"
                    )
                with col_d2:
                    st.download_button(
                        "Download diagnostics (CSV)",
                        data=diagnostics.to_csv(index=False).encode("utf-8"),
                        file_name=f"{apt}_overnight_intervals_{int(year_int)}-{int(month_int):02d}_diagnostics.csv",
                        mime="text/csv"
                    )

        st.markdown(
            f"**Notes:**\n"
            f"- Metric A counts a tail if still on-ground at **{check_hour.strftime('%H:%M')} {local_tz_name}** the following morning (night spanning the listed Date).\n"
            f"- Metric B counts a tail if on-ground **≥ {float(threshold_hours):.1f} h** within "
            f"**{night_start.strftime('%H:%M')}–{night_end.strftime('%H:%M')} {local_tz_name}** (window spans midnight if end ≤ start).\n"
            f"- Month-start assumption is **{'ON' if assume_from_month_start else 'OFF'}**.\n"
            f"- If results look empty for the first week, double-check the **day-first** toggle and that files are for the chosen month/year.\n"
        )
else:
    st.info("Upload both CSVs to configure column mapping and run the calculation.")
