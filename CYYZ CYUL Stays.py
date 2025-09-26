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
            # If it looks like we guessed wrong (single col but has separators), try next
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

def utc_to_local(series: pd.Series, local_tz: pytz.timezone) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True, infer_datetime_format=True)
    return dt.dt.tz_convert(local_tz)

def localize_naive(series: pd.Series, local_tz: pytz.timezone) -> pd.Series:
    out = []
    for v in pd.to_datetime(series, errors="coerce", infer_datetime_format=True):
        if pd.isna(v):
            out.append(pd.NaT)
        else:
            if v.tzinfo is None:
                out.append(local_tz.localize(v))
            else:
                out.append(v.astimezone(local_tz))
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
    airport = st.text_input("Airport (ICAO)", value="CYYZ").strip().upper()
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

st.caption("All calculations are performed in the selected **Local timezone** (default America/Toronto).")

# Column mapping UI once files are present
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
        tail_arr = st.selectbox("Arrivals: Tail/Registration column", arr_cols, index=arr_cols.index(default_arr_tail) if default_arr_tail in arr_cols else 0)
        arr_to_col = st.selectbox("Arrivals: To (ICAO)", arr_cols, index=arr_cols.index(default_to) if default_to in arr_cols else 0)
        arr_time_col = st.selectbox("Arrivals: Arrival time", arr_cols, index=arr_cols.index(default_arr_time) if default_arr_time in arr_cols else 0)
    with c2:
        tail_dep = st.selectbox("Departures: Tail/Registration column", dep_cols, index=dep_cols.index(default_dep_tail) if default_dep_tail in dep_cols else 0)
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

    st.subheader("5) Run")
    if st.button("Compute Overnights"):
        # ===============================
        # Parse & filter rows to airport
        # ===============================
        arr = arr_raw[arr_raw[arr_to_col].astype(str).str.upper() == airport].copy()
        dep = dep_raw[dep_raw[dep_from_col].astype(str).str.upper() == airport].copy()

        # Parse timestamps from source tz to LOCAL_TZ
        if ts_source_tz.startswith("UTC"):
            arr["arr_dt"] = utc_to_local(arr[arr_time_col], LOCAL_TZ)
            dep["dep_dt"] = utc_to_local(dep[dep_time_col], LOCAL_TZ)
        else:
            # Treat timestamps as local (naive or TZ-aware) and ensure LOCAL_TZ
            arr["arr_dt"] = localize_naive(arr[arr_time_col], LOCAL_TZ)
            dep["dep_dt"] = localize_naive(dep[dep_time_col], LOCAL_TZ)

        # Normalize tail
        arr["tail"] = arr[tail_arr].astype(str).str.strip().str.upper()
        dep["tail"] = dep[tail_dep].astype(str).str.strip().str.upper()

        # Drop missing datetimes
        arr = arr.dropna(subset=["arr_dt"])
        dep = dep.dropna(subset=["dep_dt"])

        # Month window
        start_local = LOCAL_TZ.localize(datetime(int(year_int), int(month_int), 1, 0, 0, 0))
        if month_int == 12:
            end_month_start = LOCAL_TZ.localize(datetime(int(year_int) + 1, 1, 1, 0, 0, 0))
        else:
            end_month_start = LOCAL_TZ.localize(datetime(int(year_int), int(month_int) + 1, 1, 0, 0, 0))
        end_local = end_month_start - timedelta(seconds=1)

        # Keep only events inside month (optional, helps pairing clarity)
        arr = arr[(arr["arr_dt"] >= start_local) & (arr["arr_dt"] <= end_local)].copy()
        dep = dep[(dep["dep_dt"] >= start_local) & (dep["dep_dt"] <= end_local)].copy()

        # ===============================
        # Build on-ground intervals
        # ===============================
        arr = arr.sort_values(["tail", "arr_dt"])
        dep = dep.sort_values(["tail", "dep_dt"])
        dep_map = {t: g["dep_dt"].tolist() for t, g in dep.groupby("tail")}

        intervals_by_tail = {}

        for tail, g in arr.groupby("tail"):
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

        # Tails with month departures but no month arrivals -> assume parked from month start until first departure
        tails_with_dep = set(dep["tail"].unique())
        tails_with_arr = set(arr["tail"].unique())
        for tail in tails_with_dep:
            if tail not in tails_with_arr:
                first_dep = dep[dep["tail"] == tail]["dep_dt"].min()
                if pd.notna(first_dep) and first_dep > start_local:
                    intervals_by_tail.setdefault(tail, []).append((start_local, first_dep))

        # Clip & merge overlaps
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

        # ===============================
        # Metric A: presence at check_hour
        # ===============================
        dates = pd.date_range(start_local.date(), end_local.date(), freq="D")
        rows_A = []
        for d in dates:
            check_dt = LOCAL_TZ.localize(datetime(d.year, d.month, d.day, check_hour.hour, check_hour.minute))
            present = []
            for tail, ivls in intervals_by_tail.items():
                if any(s <= check_dt <= e for s, e in ivls):
                    present.append(tail)
            rows_A.append({
                "Date": pd.to_datetime(d),
                "Overnights_A_check": len(present),
                "Tails_A": ", ".join(sorted(present))
            })
        df_A = pd.DataFrame(rows_A)

        # ===============================
        # Metric B: >= threshold within night window
        # ===============================
        night_rows = []
        for d in dates:
            win_start = LOCAL_TZ.localize(datetime(d.year, d.month, d.day, night_start.hour, night_start.minute))
            # compute window end (span midnight if needed)
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
            night_rows.append({
                "Date": pd.to_datetime(d),
                "Overnights_B_nightwindow": len(present_B),
                "Tails_B": ", ".join(sorted(present_B))
            })
        df_B = pd.DataFrame(night_rows)

        # ===============================
        # Combine + diagnostics
        # ===============================
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

        st.success("Computed!")

        st.subheader("Results")
        st.dataframe(combined, use_container_width=True)

        st.subheader("Diagnostics (on-ground intervals per tail)")
        st.caption("Use this to spot-check pairings and durations. Filter by Tail with the column tools.")
        st.dataframe(diagnostics, use_container_width=True, height=400)

        # Downloads
        detail_csv = combined.to_csv(index=False).encode("utf-8")
        diag_csv = diagnostics.to_csv(index=False).encode("utf-8")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "Download results (CSV)",
                data=detail_csv,
                file_name=f"{airport}_overnights_{int(year_int)}-{int(month_int):02d}_metrics.csv",
                mime="text/csv"
            )
        with col_d2:
            st.download_button(
                "Download diagnostics (CSV)",
                data=diag_csv,
                file_name=f"{airport}_overnight_intervals_{int(year_int)}-{int(month_int):02d}_diagnostics.csv",
                mime="text/csv"
            )

        st.markdown(
            f"**Notes:**\n"
            f"- Metric A counts a tail if on-ground at **{check_hour.strftime('%H:%M')} {local_tz_name}**.\n"
            f"- Metric B counts a tail if on-ground **≥ {float(threshold_hours):.1f} h** within "
            f"**{night_start.strftime('%H:%M')}–{night_end.strftime('%H:%M')} {local_tz_name}** (window spans midnight if end ≤ start).\n"
            f"- If a tail departs in the month without an in-month arrival, it is assumed on-ground from the **month start** until its first departure.\n"
        )
else:
    st.info("Upload both CSVs to configure column mapping and run the calculation.")
