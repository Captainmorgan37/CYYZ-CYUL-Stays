# overnights_cyyz.py
import calendar
import io
import hashlib
from collections import deque
from datetime import datetime, timedelta, time

import pandas as pd
import pytz
import streamlit as st
import plotly.graph_objects as go

# ===============================
# Page config & title
# ===============================
st.set_page_config(page_title="CYYZ Overnights Calculator", layout="wide")
st.title("CYYZ Overnights Calculator")
st.caption("Upload FL3XX arrivals/departures CSVs for a selected date range and compute overnight counts by day (two metrics).")

st.session_state.setdefault("overnights_results", None)

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


EMBRAER_PREFIXES = ("E54", "E55")
CJ_PREFIXES = ("C25",)


def df_signature(df: pd.DataFrame) -> str:
    """Return a lightweight hash representing the dataframe contents."""
    if df is None or df.empty:
        return "empty"
    normalized = df.astype(str)
    hashed = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
    return hashlib.md5(hashed).hexdigest()


def classify_aircraft_type(type_code: str) -> str:
    """Classify the aircraft type into embraer/cj/other/unknown buckets."""
    code = (type_code or "").strip().upper()
    if not code:
        return "unknown"
    if any(code.startswith(prefix) for prefix in EMBRAER_PREFIXES):
        return "embraer"
    if any(code.startswith(prefix) for prefix in CJ_PREFIXES):
        return "cj"
    return "other"


def build_results_csv(
    combined: pd.DataFrame,
    summary_counts: pd.DataFrame,
    average_counts: pd.DataFrame,
) -> bytes:
    """Create a CSV output with a clear summary section appended."""
    buffer = io.StringIO()
    buffer.write("# Detailed daily results\n")
    combined.to_csv(buffer, index=False)
    buffer.write("\n# Summary counts by aircraft category\n")
    summary_counts.to_csv(buffer, index=False)
    buffer.write("\n# Average counts per day (by metric)\n")
    average_counts.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def render_monthly_calendar_view(
    data: pd.DataFrame,
    value_col: str,
    title: str,
    key_prefix: str,
    value_label: str | None = None,
    value_formatter=None,
    months_per_view: int = 1,
) -> bool:
    """Render a navigable monthly calendar with daily values inside Streamlit."""

    if data.empty:
        return False

    df = data.copy()
    df["date"] = pd.to_datetime(df["ds"], errors="coerce").dt.date
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["date", value_col])

    if df.empty:
        return False

    df = df.groupby("date", as_index=False)[value_col].mean()

    month_options = sorted({d.replace(day=1) for d in df["date"]})
    if not month_options:
        return False

    style_key = "_monthly_calendar_style"
    if not st.session_state.get(style_key):
        st.markdown(
            """
            <style>
            .calendar-wrapper {margin-top: 0.5rem;}
            .calendar-month-label {text-align: center; font-weight: 700; font-size: 1.2rem; padding-top: 0.25rem;}
            .calendar-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; align-items: start;}
            .calendar-card {background: #ffffff08; padding: 0.5rem; border-radius: 0.75rem; border: 1px solid #e0e0e0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);}
            .calendar-table {width: 100%; border-collapse: collapse; table-layout: fixed; border-radius: 0.65rem; overflow: hidden; border: 1px solid #e0e0e0;}
            .calendar-table thead {background: #f6f6f6;}
            .calendar-table th {padding: 0.6rem 0.2rem; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: #555;}
            .calendar-table td {border: 1px solid #e9ecef; height: 5.2rem; vertical-align: top; padding: 0.35rem; position: relative; background: #ffffff;}
            .calendar-day {border: 1px solid #e9ecef;}
            .calendar-day .calendar-date {font-size: 0.9rem; font-weight: 600;}
            .calendar-day .calendar-value {font-size: 1.05rem; font-weight: 600; margin-top: 0.35rem;}
            .calendar-day.other-month {background: #fafafa; color: #b5b5b5;}
            .calendar-legend {font-size: 0.8rem; color: #6c757d; margin-top: 0.3rem;}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state[style_key] = True

    safe_prefix = "".join(c if c.isalnum() or c in "_-" else "_" for c in key_prefix)

    months_per_view = max(1, int(months_per_view))
    max_start_idx = max(0, len(month_options) - months_per_view)

    state_key = f"{safe_prefix}_month_index"
    idx = st.session_state.get(state_key, 0)
    idx = max(0, min(idx, max_start_idx))
    st.session_state[state_key] = idx

    st.subheader(title)
    nav_cols = st.columns([1, 3, 1])
    with nav_cols[0]:
        prev_clicked = st.button("◀", key=f"{safe_prefix}_prev", disabled=idx == 0)
    month_placeholder = nav_cols[1].empty()
    with nav_cols[2]:
        next_clicked = st.button("▶", key=f"{safe_prefix}_next", disabled=idx >= max_start_idx)

    if prev_clicked and idx > 0:
        idx = max(0, idx - months_per_view)
        st.session_state[state_key] = idx
    if next_clicked and idx < max_start_idx:
        idx = min(max_start_idx, idx + months_per_view)
        st.session_state[state_key] = idx

    displayed_months = month_options[idx : idx + months_per_view]
    first_label = displayed_months[0].strftime("%B %Y")
    last_label = displayed_months[-1].strftime("%B %Y")
    label = first_label if first_label == last_label else f"{first_label} – {last_label}"
    month_placeholder.markdown(f"<div class='calendar-month-label'>{label}</div>", unsafe_allow_html=True)

    cal = calendar.Calendar(firstweekday=6)
    month_weeks = {
        m: cal.monthdatescalendar(m.year, m.month)
        for m in displayed_months
    }

    value_map = {row.date: getattr(row, value_col) for row in df.itertuples()}
    month_values = []
    for m, weeks in month_weeks.items():
        for week in weeks:
            for day in week:
                if day.month != m.month:
                    continue
                val = value_map.get(day)
                if val is not None:
                    month_values.append(val)

    min_val = min(month_values) if month_values else None
    max_val = max(month_values) if month_values else None

    def default_formatter(val):
        if val is None:
            return ""
        if abs(val - round(val)) < 0.01:
            return f"{int(round(val))}"
        return f"{val:.1f}"

    fmt = value_formatter or default_formatter

    def cell_style(val):
        if val is None or min_val is None or max_val is None:
            return ""
        if max_val == min_val:
            alpha = 0.65
        else:
            alpha = 0.2 + 0.6 * (val - min_val) / (max_val - min_val)
        alpha = max(0.15, min(alpha, 0.85))
        text_color = "#0d1b2a" if alpha < 0.55 else "#ffffff"
        return f"background-color: rgba(33, 150, 243, {alpha}); color: {text_color};"

    headers = "".join(
        f"<th>{day}</th>" for day in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    )

    month_tables = []
    for m, weeks in month_weeks.items():
        rows_html = []
        for week in weeks:
            cells = []
            for day in week:
                val = value_map.get(day)
                classes = ["calendar-day"]
                style = ""
                value_html = ""
                if day.month != m.month:
                    classes.append("other-month")
                else:
                    style = cell_style(val)
                    value_html = f"<div class='calendar-value'>{fmt(val)}</div>"
                cells.append(
                    "<td class='{}' style='{}'>".format(" ".join(classes), style)
                    + f"<div class='calendar-date'>{day.day}</div>"
                    + value_html
                    + "</td>"
                )
            rows_html.append(f"<tr>{''.join(cells)}</tr>")

        table_html = (
            "<div class='calendar-card'>"
            f"<div class='calendar-month-label'>{m.strftime('%B %Y')}</div>"
            "<div class='calendar-wrapper'>"
            "<table class='calendar-table'>"
            f"<thead><tr>{headers}</tr></thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
            "</div>"
            "</div>"
        )
        month_tables.append(table_html)

    grid_template = f"repeat({months_per_view}, minmax(220px, 1fr))"
    st.markdown(
        f"<div class='calendar-grid' style=\"grid-template-columns: {grid_template};\">" + "".join(month_tables) + "</div>",
        unsafe_allow_html=True,
    )

    if value_label:
        st.caption(f"Daily values show {value_label}.")

    return True


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
col_a, col_b = st.columns([1.2, 1.2])
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
    today = datetime.now()
    default_start = datetime(today.year, today.month, 1).date()
    last_day = calendar.monthrange(today.year, today.month)[1]
    default_end = datetime(today.year, today.month, last_day).date()
    date_range = st.date_input(
        "Reporting range (start and end dates, inclusive)",
        value=(default_start, default_end),
        help="Choose the start and end date to include in the analysis."
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    if start_date > end_date:
        st.warning("Start date is after end date. Swapping them for processing.")
        start_date, end_date = end_date, start_date

local_tz_name = "America/Toronto"
LOCAL_TZ = pytz.timezone(local_tz_name)
DAYFIRST = True

st.caption(
    "Timestamps are assumed to be in UTC and converted to **America/Toronto** "
    "(day-first parsing enabled for dates like 01.08.2025)."
)

if not arr_raw.empty and not dep_raw.empty:
    arr_cols = list(arr_raw.columns)
    dep_cols = list(dep_raw.columns)

    # Sensible defaults (prefer Actual) using automatic detection
    tail_arr = pick_col(arr_cols, ["Aircraft", "Tail", "Registration", "A/C"])
    tail_dep = pick_col(dep_cols, ["Aircraft", "Tail", "Registration", "A/C"])
    arr_to_col = pick_col(arr_cols, ["To (ICAO)", "To", "Destination"])
    dep_from_col = pick_col(dep_cols, ["From (ICAO)", "From", "Origin"])
    arr_time_col = pick_col(arr_cols, ["On-Block (Act)", "On-Block (Actual)", "On-Block", "ATA", "Arrival (UTC)"])
    dep_time_col = pick_col(dep_cols, ["Off-Block (Act)", "Off-Block (Actual)", "Off-Block", "ATD", "Departure (UTC)"])
    arr_type_col = pick_col(arr_cols, ["Aircraft Type", "Type", "A/C Type", "AC Type"])
    dep_type_col = pick_col(dep_cols, ["Aircraft Type", "Type", "A/C Type", "AC Type"])

    with st.expander("Detected column mapping", expanded=False):
        mapping_rows = [
            ("Arrivals", "Tail/Registration", tail_arr),
            ("Arrivals", "To (ICAO)", arr_to_col),
            ("Arrivals", "Arrival time", arr_time_col),
            ("Arrivals", "Aircraft type", arr_type_col or "<None detected>"),
            ("Departures", "Tail/Registration", tail_dep),
            ("Departures", "From (ICAO)", dep_from_col),
            ("Departures", "Departure time", dep_time_col),
            ("Departures", "Aircraft type", dep_type_col or "<None detected>"),
        ]
        mapping_df = pd.DataFrame(mapping_rows, columns=["File", "Field", "Detected Column"])
        st.dataframe(mapping_df, hide_index=True, use_container_width=True)

    st.subheader("3) Overnight Definitions")
    check_hour = st.time_input("Metric A — On ground at (local time)", value=time(3, 0))

    st.subheader("4) Run")

    current_signature = (
        tuple(airports),
        start_date,
        end_date,
        local_tz_name,
        tail_arr,
        arr_to_col,
        arr_time_col,
        arr_type_col or "",
        tail_dep,
        dep_from_col,
        dep_time_col,
        dep_type_col or "",
        check_hour.isoformat(),
        df_signature(arr_raw),
        df_signature(dep_raw),
    )

    stored_results = st.session_state.get("overnights_results")
    if stored_results and stored_results.get("signature") != current_signature:
        st.session_state["overnights_results"] = None
        stored_results = None

    run_compute = st.button("Compute Overnights")
    if run_compute:
        if not airports:
            st.error("Enter at least one airport code (e.g., CYYZ, CYUL).")
            st.stop()

        required_columns = {
            "Arrivals: Tail/Registration": tail_arr,
            "Arrivals: To (ICAO)": arr_to_col,
            "Arrivals: Arrival time": arr_time_col,
            "Departures: Tail/Registration": tail_dep,
            "Departures: From (ICAO)": dep_from_col,
            "Departures: Departure time": dep_time_col,
        }
        missing_required = [label for label, col in required_columns.items() if not col]
        if missing_required:
            st.error(
                "Could not auto-detect the following required columns: "
                + ", ".join(missing_required)
                + ". Please check the uploaded CSV files."
            )
            st.stop()

        # ===============================
        # Parse & normalize rows
        # ===============================
        arr = arr_raw.copy()
        dep = dep_raw.copy()

        arr["airport_dest"] = arr[arr_to_col].astype(str).str.strip().str.upper()
        dep["airport_origin"] = dep[dep_from_col].astype(str).str.strip().str.upper()

        # Parse timestamps from source tz to LOCAL_TZ
        arr["arr_dt"] = parse_flexible_utc_to_local(arr[arr_time_col], LOCAL_TZ, dayfirst=DAYFIRST)
        dep["dep_dt"] = parse_flexible_utc_to_local(dep[dep_time_col], LOCAL_TZ, dayfirst=DAYFIRST)

        # Sanity check BEFORE range filter
        all_parsed = pd.concat([arr["arr_dt"].dropna(), dep["dep_dt"].dropna()])
        if not all_parsed.empty:
            start_dt_check = LOCAL_TZ.localize(datetime.combine(start_date, time(0, 0, 0)))
            end_dt_check = LOCAL_TZ.localize(datetime.combine(end_date, time(23, 59, 59)))
            range_hits = ((all_parsed >= start_dt_check) & (all_parsed <= end_dt_check)).mean()
            if range_hits < 0.2:
                st.warning(
                    "Heads up: Most parsed timestamps are NOT in the chosen reporting range. "
                    "This usually means the files cover different dates or use unexpected formats."
                )

        # Normalize tail
        arr["tail"] = arr[tail_arr].astype(str).str.strip().str.upper()
        dep["tail"] = dep[tail_dep].astype(str).str.strip().str.upper()

        if arr_type_col:
            arr["aircraft_type"] = arr[arr_type_col].astype(str).str.strip().str.upper()
        else:
            arr["aircraft_type"] = ""
        if dep_type_col:
            dep["aircraft_type"] = dep[dep_type_col].astype(str).str.strip().str.upper()
        else:
            dep["aircraft_type"] = ""

        # Drop missing datetimes
        arr = arr.dropna(subset=["arr_dt"])
        dep = dep.dropna(subset=["dep_dt"])

        # Reporting window (used for clipping/reporting, but we keep surrounding events for proper pairing)
        start_local = LOCAL_TZ.localize(datetime.combine(start_date, time(0, 0, 0)))
        end_local = LOCAL_TZ.localize(datetime.combine(end_date, time(23, 59, 59)))

        # We need to retain enough timeline past the reporting range end so that
        # checks that occur in the following morning (Metric A) still see the aircraft as
        # present. Otherwise the last day of the reporting range would always
        # appear empty because the intervals get clipped at 23:59:59.
        last_date = end_local.date()
        check_dt_last = LOCAL_TZ.localize(
            datetime(last_date.year, last_date.month, last_date.day, check_hour.hour, check_hour.minute)
        ) + timedelta(days=1)
        clip_end = max(end_local, check_dt_last)

        def compute_airport_metrics(airport_code: str):
            arr_filtered = arr[arr["airport_dest"] == airport_code].copy()
            dep_filtered = dep[dep["airport_origin"] == airport_code].copy()

            tail_type_map = {}
            for df_types in (arr_filtered, dep_filtered):
                if "aircraft_type" in df_types.columns:
                    for tail_val, type_val in zip(df_types["tail"], df_types["aircraft_type"]):
                        tail_clean = str(tail_val).strip().upper()
                        type_clean = str(type_val).strip().upper()
                        if tail_clean and type_clean and tail_clean not in tail_type_map:
                            tail_type_map[tail_clean] = type_clean

            arr_filtered = arr_filtered.sort_values(["tail", "arr_dt"])
            dep_filtered = dep_filtered.sort_values(["tail", "dep_dt"])

            arr_map = {t: g["arr_dt"].tolist() for t, g in arr_filtered.groupby("tail")}
            dep_map = {t: g["dep_dt"].tolist() for t, g in dep_filtered.groupby("tail")}

            intervals_by_tail = {}
            all_tails = sorted(set(arr_map) | set(dep_map))
            for tail in all_tails:
                arr_times = arr_map.get(tail, [])
                dep_times = dep_map.get(tail, [])
                events = [
                    (ts, 0) for ts in arr_times
                ] + [
                    (ts, 1) for ts in dep_times
                ]
                events.sort(key=lambda x: (x[0], x[1]))

                queue = deque()
                own_intervals = []
                for ts, kind in events:
                    if kind == 0:  # arrival
                        queue.append(ts)
                    else:  # departure
                        if queue and queue[0] <= ts:
                            arr_t = queue.popleft()
                            if ts > arr_t:
                                own_intervals.append((arr_t, ts))
                        # if there's no matching arrival, skip the departure

                while queue:
                    arr_t = queue.popleft()
                    if arr_t < clip_end:
                        own_intervals.append((arr_t, clip_end))

                intervals_by_tail[tail] = own_intervals

            for tail, ivls in list(intervals_by_tail.items()):
                clipped = []
                for s, e in ivls:
                    if e < start_local or s > clip_end:
                        continue
                    s2 = max(s, start_local)
                    e2 = min(e, clip_end)
                    if e2 > s2:
                        clipped.append((s2, e2))
                intervals_by_tail[tail] = merge_intervals(clipped)

            dates = pd.date_range(start_local.date(), end_local.date(), freq="D")
            rows_A = []
            for d in dates:
                check_dt = LOCAL_TZ.localize(datetime(d.year, d.month, d.day, check_hour.hour, check_hour.minute)) + timedelta(days=1)
                present = sorted([t for t, ivls in intervals_by_tail.items() if any(s <= check_dt <= e for s, e in ivls)])
                emb_tails = sorted([t for t in present if classify_aircraft_type(tail_type_map.get(t, "")) == "embraer"])
                cj_tails = sorted([t for t in present if classify_aircraft_type(tail_type_map.get(t, "")) == "cj"])
                rows_A.append({
                    "Date": pd.to_datetime(d),
                    "Overnights_A_check": len(present),
                    "Tails_A": ", ".join(present),
                    "Embraer_A_Count": len(emb_tails),
                    "Embraer_A_Tails": ", ".join(emb_tails),
                    "CJ_A_Count": len(cj_tails),
                    "CJ_A_Tails": ", ".join(cj_tails),
                })
            df_A = pd.DataFrame(rows_A)

            combined = df_A.sort_values("Date").reset_index(drop=True)

            summary_counts = combined[["Date", "Overnights_A_check", "Embraer_A_Count", "CJ_A_Count"]].rename(columns={
                "Overnights_A_check": "Total_Tails",
                "Embraer_A_Count": "Embraer_Count",
                "CJ_A_Count": "CJ_Count",
            })
            summary_counts["Metric"] = "Metric A"
            summary_counts = summary_counts[["Date", "Metric", "Total_Tails", "Embraer_Count", "CJ_Count"]]

            average_counts = summary_counts.groupby("Metric")[
                ["Total_Tails", "Embraer_Count", "CJ_Count"]
            ].mean().reset_index()
            average_counts = average_counts.rename(columns={
                "Total_Tails": "Avg_Total_Tails_per_Day",
                "Embraer_Count": "Avg_Embraer_per_Day",
                "CJ_Count": "Avg_CJ_per_Day",
            })

            diag_rows = []
            for tail, ivls in sorted(intervals_by_tail.items()):
                for s, e in ivls:
                    diag_rows.append({
                        "Tail": tail,
                        "OnGround_Start_Local": s,
                        "OnGround_End_Local": e,
                        "Hours": (e - s).total_seconds() / 3600.0,
                        "Aircraft_Type": tail_type_map.get(tail, "")
                    })
            diagnostics = pd.DataFrame(diag_rows).sort_values(["Tail", "OnGround_Start_Local"]).reset_index(drop=True)

            return combined, diagnostics, summary_counts, average_counts

        metrics = {apt: compute_airport_metrics(apt) for apt in airports}

        result_payload = {
            "metrics": metrics,
            "airports": airports,
            "signature": current_signature,
        }
        st.session_state["overnights_results"] = result_payload
        stored_results = result_payload

    if stored_results:
        airports_cached = stored_results["airports"]
        metrics_cached = stored_results["metrics"]
        if run_compute:
            st.success(f"Computed for {', '.join(airports_cached)}!")
        else:
            st.success(f"Using cached results for {', '.join(airports_cached)}.")

        st.subheader("Results")
        tabs = st.tabs([f"{apt}" for apt in airports_cached])
        for tab, apt in zip(tabs, airports_cached):
            combined, diagnostics, summary_counts, average_counts = metrics_cached[apt]
            with tab:
                st.markdown(f"### {apt}")
                st.dataframe(combined, use_container_width=True)

                st.subheader("Daily summary by aircraft category")
                st.dataframe(summary_counts, use_container_width=True)

                st.subheader("Average aircraft counts per day")
                st.dataframe(average_counts, use_container_width=True)

                st.subheader("Diagnostics (on-ground intervals per tail)")
                st.caption("Use this to spot-check pairings and durations. Filter by Tail with the column tools.")
                st.dataframe(diagnostics, use_container_width=True, height=400)

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        "Download results (CSV)",
                        data=build_results_csv(combined, summary_counts, average_counts),
                        file_name=f"{apt}_overnights_{start_date.isoformat()}_{end_date.isoformat()}_metrics.csv",
                        mime="text/csv"
                    )
                with col_d2:
                    st.download_button(
                        "Download diagnostics (CSV)",
                        data=diagnostics.to_csv(index=False).encode("utf-8"),
                        file_name=f"{apt}_overnight_intervals_{start_date.isoformat()}_{end_date.isoformat()}_diagnostics.csv",
                        mime="text/csv"
                    )

        st.markdown(
            f"**Notes:**\n"
            f"- Metric A counts a tail if still on-ground at **{check_hour.strftime('%H:%M')} {local_tz_name}** the following morning (night spanning the listed Date).\n"
            f"- Arrivals/departures just outside the reporting range are automatically considered for pairing, so include surrounding days in the uploads for best accuracy.\n"
            f"- If results look empty for the first week, confirm the uploads cover the chosen reporting range and use the expected date formats.\n"
        )
else:
    st.info("Upload both CSVs to configure column mapping and run the calculation.")


results_state = st.session_state.get("overnights_results")
cached_metrics = results_state["metrics"] if results_state else {}
cached_airports = results_state["airports"] if results_state else []



# ===============================
# Forecast Tab – Predictive Schedule
# ===============================
st.header("✈ Predictive Schedule (Experimental)")

if not arr_raw.empty and not dep_raw.empty:
    st.subheader("Historical Pattern Forecast")
    st.caption("Based on historical overnight counts since 2024, predict expected stays for upcoming days.")

    import numpy as np
    from prophet import Prophet
    import plotly.express as px

    # Let the user choose forecast length
    forecast_days = st.slider("Forecast horizon (days ahead)", 7, 180, 60, 1)

    # Build historical daily counts from combined results
    all_airports = cached_airports or airports or ["CYYZ"]
    hist_rows = []
    for apt in all_airports:
        combined, _, _, _ = cached_metrics.get(apt, (pd.DataFrame(), None, None, None))
        if not combined.empty:
            tmp = combined[["Date", "Overnights_A_check"]].copy()
            tmp["Airport"] = apt
            hist_rows.append(tmp)
    if hist_rows:
        hist_df = pd.concat(hist_rows, ignore_index=True)
    else:
        hist_df = pd.DataFrame(columns=["Date", "Overnights_A_check", "Airport"])

    if hist_df.empty:
        st.info("Run the overnight calculation first to generate historical data.")
    else:
        # Prophet expects columns named 'ds' (date) and 'y' (value)
        airport = st.selectbox("Select airport for forecast", all_airports)
        df_airport = hist_df[hist_df["Airport"] == airport][["Date", "Overnights_A_check"]].rename(
            columns={"Date": "ds", "Overnights_A_check": "y"}
        )
        df_airport = df_airport.groupby("ds").mean().reset_index()

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(df_airport)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        fig = px.line(
            forecast,
            x="ds",
            y=["yhat", "yhat_lower", "yhat_upper"],
            labels={"ds": "Date", "value": "Predicted Overnights"},
            title=f"{airport} Predicted Overnights ({forecast_days}-day horizon)"
        )
        st.plotly_chart(fig, use_container_width=True)

        future_forecast = forecast[forecast["ds"] > df_airport["ds"].max()].tail(forecast_days)
        if future_forecast.empty:
            future_forecast = forecast.tail(forecast_days)

        calendar_metric = st.radio(
            "Calendar values",
            ["Mean (yhat)", "Upper limit (yhat_upper)"],
            horizontal=True,
            key=f"forecast_calendar_toggle_{airport}",
        )
        calendar_column = "yhat" if calendar_metric.startswith("Mean") else "yhat_upper"
        calendar_label = "predicted mean overnights" if calendar_column == "yhat" else "upper bound overnights"

        render_monthly_calendar_view(
            future_forecast[["ds", "yhat", "yhat_upper"]],
            value_col=calendar_column,
            title="Forecast calendar view",
            key_prefix=f"forecast_calendar_{airport}",
            value_label=calendar_label,
            months_per_view=3,
        )

        # Summary table
        st.subheader("Forecast Summary")
        st.dataframe(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days),
            use_container_width=True
        )

        st.caption(
            "Uses a Prophet model with automatic detection of weekly and seasonal patterns. "
            "Values are mean ± 80% confidence interval."
        )
else:
    st.info("Upload data and run calculations to enable the forecast feature.")


# ===============================
# Forecast Tab – Predictive Movements (Arrivals, Departures & Total Load)
# ===============================
st.header("📈 Predictive Movements (Arrivals, Departures & Total Load)")

if not arr_raw.empty and not dep_raw.empty:
    st.subheader("Daily Movement Forecast")
    st.caption(
        "Predict expected arrivals, departures, and total daily movements based on historical data. "
        "Use the toggle to include historical data in the forecast chart."
    )

    import numpy as np
    from prophet import Prophet
    import plotly.graph_objects as go

    forecast_days = st.slider(
        "Forecast horizon (days ahead)", 7, 180, 60, 1, key="mov_forecast_days"
    )

    # --- Build daily counts ---
    arr = arr_raw.copy()
    dep = dep_raw.copy()

    # Dynamic column mapping for uploaded CSVs
    arr_time_col = pick_col(arr.columns, ["On-Block (Act)", "Arrival_Time", "On Block", "OnBlock"])
    dep_time_col = pick_col(dep.columns, ["Off-Block (Act)", "Departure_Time", "Off Block", "OffBlock"])
    arr_apt_col = pick_col(arr.columns, ["To (ICAO)", "Destination", "To"])
    dep_apt_col = pick_col(dep.columns, ["From (ICAO)", "Origin", "From"])

    arr["date"] = pd.to_datetime(arr[arr_time_col], errors="coerce", dayfirst=DAYFIRST).dt.date
    dep["date"] = pd.to_datetime(dep[dep_time_col], errors="coerce", dayfirst=DAYFIRST).dt.date
    arr[arr_apt_col] = arr[arr_apt_col].astype(str).str.upper().str.strip()
    dep[dep_apt_col] = dep[dep_apt_col].astype(str).str.upper().str.strip()

    # Count arrivals and departures per airport per day
    arrivals_daily = arr.groupby(["date", arr_apt_col]).size().reset_index(name="Arrivals")
    departures_daily = dep.groupby(["date", dep_apt_col]).size().reset_index(name="Departures")

    # Merge arrivals and departures
    daily_mov = pd.merge(
        arrivals_daily,
        departures_daily,
        how="outer",
        left_on=["date", arr_apt_col],
        right_on=["date", dep_apt_col],
    ).fillna(0)

    daily_mov["Airport"] = daily_mov[arr_apt_col].combine_first(daily_mov[dep_apt_col])
    daily_mov["Arrivals"] = daily_mov["Arrivals"].astype(int)
    daily_mov["Departures"] = daily_mov["Departures"].astype(int)
    daily_mov["Ground_Load_Index"] = daily_mov["Arrivals"] + daily_mov["Departures"]
    daily_mov = daily_mov[["date", "Airport", "Arrivals", "Departures", "Ground_Load_Index"]]

    # Filter to selected airports
    daily_mov = daily_mov[daily_mov["Airport"].isin(airports)]

    if daily_mov.empty:
        st.warning("No matching airport data found in selected files.")
    else:
        airport = st.selectbox(
            "Select airport for forecast",
            sorted(daily_mov["Airport"].unique()),
            key="mov_airport",
        )
        metric = st.selectbox(
            "Select metric to forecast",
            ["Arrivals", "Departures", "Ground_Load_Index"],
            key="mov_metric",
        )
        show_historical = st.checkbox("Show historical overlay", value=True, key="mov_show_hist")

        df = daily_mov[daily_mov["Airport"] == airport][["date", "Arrivals", "Departures", "Ground_Load_Index"]]
        df = df.groupby("date").sum().reset_index()

        # Build Prophet model
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        df_train = df.rename(columns={"date": "ds", metric: "y"})
        m.fit(df_train[["ds", "y"]])
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)

        # --- Plotly chart ---
        fig = go.Figure()

        # Forecast band
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                name="Upper CI",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat_lower"],
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                name="Lower CI",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat"],
                mode="lines",
                name=f"{metric} Forecast",
                line=dict(width=3),
            )
        )

        # Optional historical overlay
        if show_historical:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["Arrivals"],
                    name="Historical Arrivals",
                    mode="lines",
                    line=dict(dash="dot", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["Departures"],
                    name="Historical Departures",
                    mode="lines",
                    line=dict(dash="dot", width=2),
                )
            )
            if metric == "Ground_Load_Index":
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df["Ground_Load_Index"],
                        name="Historical Ground Load",
                        mode="lines",
                        line=dict(dash="dot", width=2),
                    )
                )

        fig.update_layout(
            title=f"{airport} {metric} Forecast ({forecast_days}-day horizon)",
            xaxis_title="Date",
            yaxis_title=f"{metric} per Day",
            legend_title="Series",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        last_history_date = pd.to_datetime(df_train["ds"].max())
        future_forecast = forecast[pd.to_datetime(forecast["ds"]) > last_history_date].tail(
            forecast_days
        )
        if future_forecast.empty:
            future_forecast = forecast.tail(forecast_days)

        movement_calendar_metric = st.radio(
            "Calendar values",
            ["Mean (yhat)", "Upper limit (yhat_upper)"],
            horizontal=True,
            key=f"movement_calendar_toggle_{airport}_{metric}",
        )
        movement_calendar_column = (
            "yhat" if movement_calendar_metric.startswith("Mean") else "yhat_upper"
        )
        movement_calendar_label = (
            f"predicted {metric.replace('_', ' ').lower()}" if movement_calendar_column == "yhat" else f"upper bound for {metric.replace('_', ' ').lower()}"
        )

        render_monthly_calendar_view(
            future_forecast[["ds", "yhat", "yhat_upper"]],
            value_col=movement_calendar_column,
            title=f"{airport} {metric} forecast — calendar view",
            key_prefix=f"movement_calendar_{airport}_{metric}",
            value_label=movement_calendar_label,
            months_per_view=3,
        )

        # Summary table
        st.subheader("Forecast Summary")
        st.dataframe(
            forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days),
            use_container_width=True,
        )

        st.caption(
            "Forecast generated with Prophet using weekly and yearly seasonality. "
            "Confidence interval represents ±80%. "
            "Ground Load Index = Arrivals + Departures. Toggle the historical overlay for full movement context."
        )
else:
    st.info("Upload both CSV files to enable movement forecasting.")
