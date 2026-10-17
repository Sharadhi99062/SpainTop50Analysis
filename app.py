import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SPAIN TOP 50 SONGS ANALYSIS DASHBOARD
# Required file in the same repository: Atlantic_Spain.csv
# ============================================================

st.set_page_config(
    page_title="Spain Top 50 Songs Analysis",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spain Top 50 Songs Analysis")
st.subheader("Content Maturity, Release Lifecycle & Playlist Rotation")

STAGE_ORDER = [
    "New Entry",
    "Growth Phase",
    "Peak Phase",
    "Mature Phase",
    "Decline Phase"
]


# ----------------------- LOAD DATA ---------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Atlantic_Spain.csv")

    required = [
        "date", "position", "song", "artist", "popularity",
        "duration_ms", "album_type", "total_tracks", "is_explicit"
    ]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required column(s): " + ", ".join(missing)
        )

    # The project CSV uses day-month-year dates. If some rows use another
    # date format, try pandas' general parser for rows that failed.
    parsed = pd.to_datetime(data["date"], format="%d-%m-%Y", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            data.loc[remaining, "date"], errors="coerce"
        )
    data["date"] = parsed

    for col in ["song", "artist", "album_type"]:
        data[col] = data[col].astype("string").str.strip()

    data["album_type"] = data["album_type"].str.lower()

    for col in ["position", "popularity", "duration_ms", "total_tracks"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    explicit_map = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False
    }
    data["is_explicit"] = (
        data["is_explicit"].astype("string").str.strip().str.lower()
        .map(explicit_map)
    )

    data = data.dropna(
        subset=["date", "song", "artist", "position"]
    ).copy()
    data = data.drop_duplicates().copy()

    data["song_id"] = (
        data["song"].astype("string").str.lower().str.strip()
        + " | "
        + data["artist"].astype("string").str.lower().str.strip()
    )
    data["duration_minutes"] = data["duration_ms"] / 60000

    # Build lifecycle labels using the full dataset BEFORE dashboard filters.
    data = data.sort_values(["song_id", "date", "position"]).copy()
    data["entry_date"] = data.groupby("song_id")["date"].transform("min")
    data["last_seen_date"] = data.groupby("song_id")["date"].transform("max")
    data["days_since_entry"] = (
        data["date"] - data["entry_date"]
    ).dt.days
    data["previous_position"] = data.groupby("song_id")["position"].shift(1)

    # A stable rolling window uses the previous seven observations for each song.
    data["top10_flag"] = (data["position"] <= 10).astype(int)
    data["top10_observations_last7"] = (
        data.groupby("song_id")["top10_flag"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )

    data["lifecycle_stage"] = "Mature Phase"

    # First seven calendar days after first appearance.
    new_mask = data["days_since_entry"] <= 7
    data.loc[new_mask, "lifecycle_stage"] = "New Entry"

    # Peak: sustained Top 10 presence among the last seven observations.
    peak_mask = (
        (data["days_since_entry"] > 7)
        & (data["top10_observations_last7"] >= 0.70)
    )
    data.loc[peak_mask, "lifecycle_stage"] = "Peak Phase"

    # Growth/decline labels apply outside the Top 10 after the new-entry period.
    growth_mask = (
        (data["days_since_entry"] > 7)
        & (data["position"] > 10)
        & data["previous_position"].notna()
        & (data["position"] < data["previous_position"])
        & ~peak_mask
    )
    data.loc[growth_mask, "lifecycle_stage"] = "Growth Phase"

    decline_mask = (
        (data["days_since_entry"] > 7)
        & (data["position"] > 10)
        & data["previous_position"].notna()
        & (data["position"] > data["previous_position"])
        & ~peak_mask
    )
    data.loc[decline_mask, "lifecycle_stage"] = "Decline Phase"

    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Cannot find Atlantic_Spain.csv. Upload it to the same GitHub "
        "repository and folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The dataset has no usable records after cleaning.")
    st.stop()


# ----------------------- SIDEBAR FILTERS ---------------------
st.sidebar.header("Dashboard Filters")
st.sidebar.caption("Filters affect the records and charts shown below.")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

# English-formatted date selection
col1, col2 = st.sidebar.columns(2)

with col1:
    start_date_text = st.text_input(
        "Start Date (YYYY-MM-DD)",
        value=str(min_date),
        key="start_date_english"
    )

with col2:
    end_date_text = st.text_input(
        "End Date (YYYY-MM-DD)",
        value=str(max_date),
        key="end_date_english"
    )

try:
    start_date = pd.to_datetime(
        start_date_text, format="%Y-%m-%d"
    ).date()

    end_date = pd.to_datetime(
        end_date_text, format="%Y-%m-%d"
    ).date()

    if start_date < min_date or end_date > max_date:
        st.sidebar.error(
            f"Choose dates between {min_date} and {max_date}."
        )
        st.stop()

    if start_date > end_date:
        st.sidebar.error(
            "Start Date must be before End Date."
        )
        st.stop()

except ValueError:
    st.sidebar.error(
        "Enter dates in YYYY-MM-DD format, for example 2025-11-27."
    )
    st.stop()
content_choice = st.sidebar.radio(
    "Explicit Content",
    ["All", "Clean only", "Explicit only"],
    index=0
)

album_types = sorted(df["album_type"].dropna().unique().tolist())
selected_album_types = st.sidebar.multiselect(
    "Album Type",
    options=album_types,
    default=album_types
)

selected_stages = st.sidebar.multiselect(
    "Lifecycle Stage",
    options=STAGE_ORDER,
    default=STAGE_ORDER
)

# Filter in one predictable order. Lifecycle labels were already calculated
# from full song history above.
date_mask = df["date"].dt.date.between(start_date, end_date)
date_filtered = df.loc[date_mask].copy()

if content_choice == "Clean only":
    date_filtered = date_filtered[date_filtered["is_explicit"] == False].copy()
elif content_choice == "Explicit only":
    date_filtered = date_filtered[date_filtered["is_explicit"] == True].copy()

date_filtered = date_filtered[
    date_filtered["album_type"].isin(selected_album_types)
].copy()

stage_counts = (
    date_filtered["lifecycle_stage"]
    .value_counts()
    .reindex(STAGE_ORDER, fill_value=0)
)

filtered = date_filtered[
    date_filtered["lifecycle_stage"].isin(selected_stages)
].copy()

if date_filtered.empty:
    st.warning("No records match the selected date, content, and album filters.")
    st.stop()

if not selected_stages:
    st.warning("Select at least one lifecycle stage in the sidebar.")
    st.stop()

if filtered.empty:
    st.warning("No records match the selected lifecycle stages.")
    st.stop()


# ----------------------- SONG-LEVEL TABLES -------------------
# Selected-period summary, so the numbers match the chosen date range.
lifecycle = filtered.groupby("song_id").agg(
    song=("song", "first"),
    artist=("artist", "first"),
    entry_date=("entry_date", "first"),
    last_seen_in_selected_period=("date", "max"),
    observed_playlist_days=("date", "nunique"),
    peak_position=("position", "min"),
    average_position=("position", "mean"),
    average_popularity=("popularity", "mean"),
    duration_ms=("duration_ms", "median"),
    total_tracks=("total_tracks", "first"),
    is_explicit=("is_explicit", "first"),
    album_type=("album_type", "first"),
    lifecycle_stage=("lifecycle_stage", "last")
).reset_index()

# Peak position date from each song's full available history.
peak_dates = (
    df.sort_values(["song_id", "position", "date"])
    .drop_duplicates("song_id")[["song_id", "date"]]
    .rename(columns={"date": "peak_position_date"})
)
lifecycle = lifecycle.merge(peak_dates, on="song_id", how="left")
lifecycle["days_to_peak_position"] = (
    lifecycle["peak_position_date"] - lifecycle["entry_date"]
).dt.days
lifecycle["duration_minutes"] = lifecycle["duration_ms"] / 60000


# ----------------------- PLAYLIST ROTATION -------------------
# Churn is calculated independently of the lifecycle-stage filter so selecting
# one stage does not falsely change the playlist's entry/exit counts.
rotation_source = df[
    df["date"].dt.date.between(start_date, end_date)
].copy()

if content_choice == "Clean only":
    rotation_source = rotation_source[
        rotation_source["is_explicit"] == False
    ].copy()
elif content_choice == "Explicit only":
    rotation_source = rotation_source[
        rotation_source["is_explicit"] == True
    ].copy()

rotation_source = rotation_source[
    rotation_source["album_type"].isin(selected_album_types)
].copy()

daily_sets = (
    rotation_source.groupby("date")["song_id"]
    .apply(set)
    .sort_index()
)

rotation_rows = []
previous_songs = None
for day, current_songs in daily_sets.items():
    if previous_songs is None:
        entries = 0
        exits = 0
        retained = len(current_songs)
        churn_pct = np.nan
        retention_pct = np.nan
    else:
        entries = len(current_songs - previous_songs)
        exits = len(previous_songs - current_songs)
        retained = len(current_songs & previous_songs)
        churn_pct = (
            (entries + exits) / (2 * max(len(current_songs), len(previous_songs))) * 100
            if max(len(current_songs), len(previous_songs)) else np.nan
        )
        retention_pct = (
            retained / len(previous_songs) * 100 if previous_songs else np.nan
        )

    rotation_rows.append({
        "date": day,
        "playlist_song_count": len(current_songs),
        "new_entries": entries,
        "exits": exits,
        "songs_retained": retained,
        "churn_percent": churn_pct,
        "retention_percent": retention_pct
    })
    previous_songs = current_songs

rotation = pd.DataFrame(rotation_rows)


# ----------------------- OVERVIEW KPIs -----------------------
st.header("1. Overview")
c1, c2, c3, c4 = st.columns(4)

c1.metric("Unique Songs", f"{lifecycle['song_id'].nunique():,}")
c2.metric(
    "Average Playlist Days",
    f"{lifecycle['observed_playlist_days'].mean():.2f}"
    if not lifecycle.empty else "N/A"
)
mean_peak_days = lifecycle["days_to_peak_position"].mean()
c3.metric(
    "Average Entry-to-Peak Time",
    f"{mean_peak_days:.2f} days" if pd.notna(mean_peak_days) else "N/A"
)
mean_churn = rotation["churn_percent"].dropna().mean()
c4.metric(
    "Average Playlist Churn",
    f"{mean_churn:.2f}%" if pd.notna(mean_churn) else "N/A"
)

c5, c6, c7, c8 = st.columns(4)
mean_retention = rotation["retention_percent"].dropna().mean()
c5.metric(
    "Average Retention",
    f"{mean_retention:.2f}%" if pd.notna(mean_retention) else "N/A"
)
c6.metric(
    "Average Popularity",
    f"{lifecycle['average_popularity'].mean():.2f}"
    if lifecycle["average_popularity"].notna().any() else "N/A"
)
c7.metric(
    "Explicit Songs",
    f"{lifecycle.loc[lifecycle['is_explicit'] == True, 'song_id'].nunique():,}"
)
c8.metric(
    "Clean Songs",
    f"{lifecycle.loc[lifecycle['is_explicit'] == False, 'song_id'].nunique():,}"
)

st.caption(
    "Lifecycle labels are rule-based analytical categories. The date range "
    "controls displayed records; stage labels use each song's full available history."
)


# ----------------------- LIFECYCLE VISUALIZER ----------------
st.header("2. Song Lifecycle Analysis")

with st.expander("View song lifecycle summary table", expanded=True):
    st.dataframe(
        lifecycle.sort_values("observed_playlist_days", ascending=False),
        use_container_width=True,
        hide_index=True
    )

st.subheader("Lifecycle Stage Distribution")
stage_chart = stage_counts.rename_axis("Lifecycle Stage").reset_index(name="Record Count")
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(stage_chart["Lifecycle Stage"], stage_chart["Record Count"])
ax.set_ylabel("Playlist Records")
ax.set_xlabel("Lifecycle Stage")
ax.tick_params(axis="x", rotation=25)
ax.set_title("Distribution of Lifecycle Stages in Selected Period")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Song Lifecycle Timeline")
timeline_pool = filtered[["song_id", "song", "artist"]].drop_duplicates()
timeline_pool = timeline_pool.sort_values(["artist", "song"])
if timeline_pool.empty:
    st.info("No songs are available for the timeline with the current filters.")
else:
    label_map = {
        row["song_id"]: f'{row["song"]} — {row["artist"]}'
        for _, row in timeline_pool.iterrows()
    }
    chosen_song = st.selectbox(
        "Select a song to view its playlist ranking over time",
        options=timeline_pool["song_id"].tolist(),
        format_func=lambda value: label_map.get(value, value)
    )
    timeline = filtered[filtered["song_id"] == chosen_song].sort_values("date")
    if not timeline.empty:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(timeline["date"], timeline["position"], marker="o")
        ax.set_title(f'Playlist Ranking Over Time: {timeline.iloc[0]["song"]}')
        ax.set_xlabel("Date")
        ax.set_ylabel("Position (1 = highest rank)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        plt.xticks(rotation=35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

st.subheader("Lifecycle Stage Comparison")
stage_comparison = date_filtered.groupby("lifecycle_stage").agg(
    records=("song_id", "size"),
    unique_songs=("song_id", "nunique"),
    average_popularity=("popularity", "mean"),
    average_position=("position", "mean")
).reindex(STAGE_ORDER).reset_index()
st.dataframe(stage_comparison.round(2), use_container_width=True, hide_index=True)


# ----------------------- ENTRY / EXIT AND CHURN ---------------
st.header("3. Playlist Rotation and Churn Analytics")
left, right = st.columns(2)

with left:
    st.subheader("Daily Entries vs Exits")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rotation["date"], rotation["new_entries"], marker="o", label="New entries")
    ax.plot(rotation["date"], rotation["exits"], marker="o", label="Exits")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of songs")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.xticks(rotation=35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Daily Churn vs Retention")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rotation["date"], rotation["churn_percent"], marker="o", label="Churn (%)")
    ax.plot(rotation["date"], rotation["retention_percent"], marker="o", label="Retention (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.xticks(rotation=35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Monthly Playlist Rotation")
if not rotation.empty:
    monthly = rotation.set_index("date").resample("MS").agg(
        total_entries=("new_entries", "sum"),
        total_exits=("exits", "sum"),
        average_churn_percent=("churn_percent", "mean"),
        average_retention_percent=("retention_percent", "mean")
    )
    st.bar_chart(monthly[["total_entries", "total_exits"]])
    st.dataframe(monthly.round(2), use_container_width=True)


# ----------------------- EXPLICIT VS CLEAN -------------------
st.header("4. Content Maturity: Explicit vs Clean")
explicit_rows = lifecycle.copy()
explicit_rows["content_type"] = explicit_rows["is_explicit"].map(
    {True: "Explicit", False: "Clean"}
).fillna("Unknown")

explicit_summary = explicit_rows.groupby("content_type").agg(
    unique_songs=("song_id", "nunique"),
    average_playlist_days=("observed_playlist_days", "mean"),
    average_peak_position=("peak_position", "mean"),
    average_popularity=("average_popularity", "mean"),
    average_days_to_peak=("days_to_peak_position", "mean")
).reset_index()
st.dataframe(explicit_summary.round(2), use_container_width=True, hide_index=True)

if not explicit_summary.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(explicit_summary["content_type"], explicit_summary["average_playlist_days"])
    ax.set_title("Average Playlist Days: Explicit vs Clean")
    ax.set_ylabel("Average days in selected period")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------- ALBUM TYPE AND SIZE -----------------
st.header("5. Single vs Album Analysis")
album_summary = lifecycle.groupby("album_type", dropna=False).agg(
    unique_songs=("song_id", "nunique"),
    average_playlist_days=("observed_playlist_days", "mean"),
    average_peak_position=("peak_position", "mean"),
    average_popularity=("average_popularity", "mean"),
    average_days_to_peak=("days_to_peak_position", "mean")
).reset_index()
st.dataframe(album_summary.round(2), use_container_width=True, hide_index=True)

if not album_summary.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(album_summary["album_type"].astype(str), album_summary["average_playlist_days"])
    ax.set_title("Average Playlist Days by Album Type")
    ax.set_xlabel("Album type")
    ax.set_ylabel("Average days")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Album Size vs Playlist Longevity")
size_source = lifecycle.copy()
size_source["total_tracks"] = pd.to_numeric(size_source["total_tracks"], errors="coerce")
size_source["album_size_group"] = pd.cut(
    size_source["total_tracks"],
    bins=[0, 1, 5, 10, 20, np.inf],
    labels=["1 Track", "2–5 Tracks", "6–10 Tracks", "11–20 Tracks", "More than 20 Tracks"],
    include_lowest=True
)
size_summary = size_source.groupby("album_size_group", observed=False).agg(
    unique_songs=("song_id", "nunique"),
    average_playlist_days=("observed_playlist_days", "mean"),
    average_popularity=("average_popularity", "mean")
).reset_index()
st.dataframe(size_summary.round(2), use_container_width=True, hide_index=True)
st.bar_chart(size_summary.set_index("album_size_group")["average_playlist_days"])


# ----------------------- POPULARITY / DURATION ---------------
st.header("6. Popularity and Duration Analysis")
left, right = st.columns(2)

with left:
    st.subheader("Popularity vs Playlist Days")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(lifecycle["average_popularity"], lifecycle["observed_playlist_days"], alpha=0.65)
    ax.set_xlabel("Average popularity")
    ax.set_ylabel("Observed playlist days")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    correlation = lifecycle[["average_popularity", "observed_playlist_days"]].corr().iloc[0, 1]
    st.write(f"Correlation: {correlation:.3f}" if pd.notna(correlation) else "Correlation unavailable")

with right:
    st.subheader("Duration vs Playlist Days")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(lifecycle["duration_minutes"], lifecycle["observed_playlist_days"], alpha=0.65)
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Observed playlist days")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    correlation = lifecycle[["duration_minutes", "observed_playlist_days"]].corr().iloc[0, 1]
    st.write(f"Correlation: {correlation:.3f}" if pd.notna(correlation) else "Correlation unavailable")


# ----------------------- DAILY DATA VALIDATION ----------------
st.header("7. Daily Record Validation")
daily_validation = df.groupby("date").agg(
    record_count=("song_id", "size"),
    unique_songs=("song_id", "nunique"),
    unique_positions=("position", "nunique"),
    minimum_position=("position", "min"),
    maximum_position=("position", "max")
).reset_index()
daily_validation["validation_status"] = np.where(
    daily_validation["record_count"] == 50,
    "Valid",
    "Check Required"
)
issues = daily_validation[daily_validation["validation_status"] != "Valid"]
v1, v2, v3 = st.columns(3)
v1.metric("Playlist Dates", f"{len(daily_validation):,}")
v2.metric("Dates with 50 Records", f"{(daily_validation['record_count'] == 50).sum():,}")
v3.metric("Dates Requiring Investigation", f"{len(issues):,}")
if not issues.empty:
    st.warning(
        "Some dates do not contain exactly 50 records. Investigate these records; "
        "the dashboard does not automatically delete or force records to 50."
    )
    st.dataframe(issues, use_container_width=True, hide_index=True)
with st.expander("View all daily validation results"):
    st.dataframe(daily_validation, use_container_width=True, hide_index=True)


# ----------------------- DOWNLOAD RESULTS ---------------------
st.header("8. Download Results")
st.download_button(
    "Download Song Lifecycle Summary",
    data=lifecycle.to_csv(index=False).encode("utf-8"),
    file_name="spain_top50_song_lifecycle.csv",
    mime="text/csv"
)
st.download_button(
    "Download Filtered Playlist Records",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="spain_top50_filtered_records.csv",
    mime="text/csv"
)
st.download_button(
    "Download Daily Rotation Metrics",
    data=rotation.to_csv(index=False).encode("utf-8"),
    file_name="spain_top50_daily_rotation.csv",
    mime="text/csv"
)
st.download_button(
    "Download Daily Validation Report",
    data=daily_validation.to_csv(index=False).encode("utf-8"),
    file_name="spain_top50_daily_validation.csv",
    mime="text/csv"
)


