import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Spain Top 50 Songs Analysis",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spain Top 50 Songs Analysis")
st.subheader(
    "Content Maturity, Release Lifecycle & Playlist Rotation"
)

st.write(
    "Analysis of Spain's Top 50 playlist for "
    "Atlantic Recording Corporation."
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("Atlantic_Spain.csv")

    data["date"] = pd.to_datetime(
        data["date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    data["song"] = data["song"].astype(str).str.strip()
    data["artist"] = data["artist"].astype(str).str.strip()

    data["album_type"] = (
        data["album_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["is_explicit"] = (
        data["is_explicit"]
        .astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False
        })
    )

    data = data.dropna(
        subset=["date", "song", "artist", "position"]
    )

    data["song_id"] = (
        data["song"].str.lower()
        + " | "
        + data["artist"].str.lower()
    )

    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Atlantic_Spain.csv was not found. "
        "Upload the dataset into the same folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

st.sidebar.caption(
    "Select a date range and content categories."
)

# --------------------------------------------------
# DATE RANGE FILTER
# --------------------------------------------------

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = min_date
    end_date = max_date

content_choice = st.sidebar.radio(
    "Explicit Content",
    ["All", "Clean only", "Explicit only"],
    index=0
)

# Apply the explicit-content filter

if content_choice == "Clean only":
    filtered = filtered[filtered["is_explicit"] == False]

elif content_choice == "Explicit only":
    filtered = filtered[filtered["is_explicit"] == True]

# Album Type Filter
all_album_types = sorted(
    df["album_type"].dropna().unique().tolist()
)

album_options = st.sidebar.multiselect(
    "Album Type",
    options=all_album_types,
    default=all_album_types
)

filtered = filtered[
    filtered["album_type"].isin(album_options)
]

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# --------------------------------------------------
# LIFECYCLE STAGE CLASSIFICATION AND FILTER
# --------------------------------------------------

# IMPORTANT:
# Calculate each song's entry date using the complete dataset,
# not just the selected date range.

stage_data = df.copy()

stage_data = stage_data.sort_values(
    ["song_id", "date"]
).copy()

# First appearance in the complete dataset
stage_data["entry_date"] = (
    stage_data.groupby("song_id")["date"]
    .transform("min")
)

# Days since the song first appeared
stage_data["days_since_entry"] = (
    stage_data["date"] - stage_data["entry_date"]
).dt.days

# Previous playlist position
stage_data["previous_position"] = (
    stage_data.groupby("song_id")["position"]
    .shift(1)
)

# Default lifecycle stage
stage_data["lifecycle_stage"] = "Mature Phase"

# 1. New Entry: first seven days
stage_data.loc[
    stage_data["days_since_entry"] <= 7,
    "lifecycle_stage"
] = "New Entry"

# 2. Peak Phase: position 1–10 after the first seven days
peak_mask = (
    (stage_data["days_since_entry"] > 7)
    & (stage_data["position"] <= 10)
)

stage_data.loc[
    peak_mask,
    "lifecycle_stage"
] = "Peak Phase"

# 3. Growth Phase: ranking improves
growth_mask = (
    (stage_data["days_since_entry"] > 7)
    & (stage_data["position"] > 10)
    & stage_data["previous_position"].notna()
    & (
        stage_data["position"]
        < stage_data["previous_position"]
    )
)

stage_data.loc[
    growth_mask,
    "lifecycle_stage"
] = "Growth Phase"

# 4. Decline Phase: ranking becomes worse
decline_mask = (
    (stage_data["days_since_entry"] > 7)
    & (stage_data["position"] > 10)
    & stage_data["previous_position"].notna()
    & (
        stage_data["position"]
        > stage_data["previous_position"]
    )
)

stage_data.loc[
    decline_mask,
    "lifecycle_stage"
] = "Decline Phase"

# Lifecycle stage options
stage_options = [
    "New Entry",
    "Growth Phase",
    "Peak Phase",
    "Mature Phase",
    "Decline Phase"
]

# Apply date range to the complete-history classification
date_filtered = stage_data[
    stage_data["date"].dt.date.between(
        start_date,
        end_date
    )
].copy()

# Apply explicit-content filter
if content_choice == "Clean only":
    date_filtered = date_filtered[
        date_filtered["is_explicit"] == False
    ]
elif content_choice == "Explicit only":
    date_filtered = date_filtered[
        date_filtered["is_explicit"] == True
    ]

# Apply album type filter
date_filtered = date_filtered[
    date_filtered["album_type"].isin(album_options)
].copy()

# Calculate distribution BEFORE applying the stage selector
stage_counts = (
    date_filtered["lifecycle_stage"]
    .value_counts()
    .reindex(stage_options, fill_value=0)
)

# Lifecycle stage selector
selected_stages = st.sidebar.multiselect(
    "Lifecycle Stage",
    options=stage_options,
    default=stage_options
)

if not selected_stages:
    st.warning("Select at least one lifecycle stage.")
    st.stop()

# Apply selected stages to the remaining analyses
filtered = date_filtered[
    date_filtered["lifecycle_stage"].isin(
        selected_stages
    )
].copy()

if filtered.empty:
    st.warning(
        "No records match the selected filters."
    )
    st.stop()

# --------------------------------------------------
# SONG LIFECYCLE
# --------------------------------------------------

lifecycle = filtered.groupby("song_id").agg(
    song=("song", "first"),
    artist=("artist", "first"),
    entry_date=("date", "min"),
    last_seen_date=("date", "max"),
    observed_playlist_days=("date", "nunique"),
    peak_position=("position", "min"),
    average_position=("position", "mean"),
    average_popularity=("popularity", "mean"),
    duration_ms=("duration_ms", "first"),
    is_explicit=("is_explicit", "first"),
    album_type=("album_type", "first")
).reset_index()

peak_rows = (
    filtered.sort_values(["song_id", "position", "date"])
    .drop_duplicates("song_id")
    [["song_id", "date"]]
    .rename(columns={"date": "peak_date"})
)

lifecycle = lifecycle.merge(
    peak_rows,
    on="song_id",
    how="left"
)

lifecycle["time_to_peak_days"] = (
    lifecycle["peak_date"] - lifecycle["entry_date"]
).dt.days

lifecycle["duration_minutes"] = (
    lifecycle["duration_ms"] / 60000
)

# --------------------------------------------------
# DAILY PLAYLIST ROTATION
# --------------------------------------------------

daily_songs = (
    filtered.groupby("date")["song_id"]
    .apply(set)
    .sort_index()
)

churn_records = []
previous_songs = None

for date, current_songs in daily_songs.items():

    if previous_songs is None:
        entries = 0
        exits = 0
    else:
        entries = len(current_songs - previous_songs)
        exits = len(previous_songs - current_songs)

    churn_records.append({
        "date": date,
        "new_entries": entries,
        "exits": exits,
        "songs_retained": (
            len(current_songs & previous_songs)
            if previous_songs is not None
            else 0
        )
    })

    previous_songs = current_songs

churn_df = pd.DataFrame(churn_records)

# --------------------------------------------------
# OVERVIEW KPIs
# --------------------------------------------------

st.header("1. Overview")

# Core dashboard metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Unique Songs",
    f"{lifecycle['song_id'].nunique():,}"
)

col2.metric(
    "Average Days on Playlist",
    f"{lifecycle['observed_playlist_days'].mean():.2f}"
)

average_entry_to_peak = lifecycle["time_to_peak_days"].mean()
col3.metric(
    "Entry-to-Peak Time",
    f"{average_entry_to_peak:.2f} days" if pd.notna(average_entry_to_peak) else "N/A"
)

if len(churn_df) > 1:
    average_churn = (churn_df["new_entries"].iloc[1:].mean() / 50) * 100
    average_retention = (churn_df["songs_retained"].iloc[1:].mean() / 50) * 100
else:
    average_churn = 0
    average_retention = 0

col4.metric(
    "Playlist Churn Rate",
    f"{average_churn:.2f}%"
)

st.subheader(" ")
kpi5, kpi6, kpi7 = st.columns(3)

kpi5.metric(
    "Retention Stability Index",
    f"{average_retention:.2f}%"
)


# Project-defined relative longevity index: Clean songs are the 100-point reference.
explicit_kpi_data = lifecycle.dropna(subset=["is_explicit"]).copy()
clean_avg_days = explicit_kpi_data.loc[
    explicit_kpi_data["is_explicit"] == False, "observed_playlist_days"
].mean()
explicit_avg_days = explicit_kpi_data.loc[
    explicit_kpi_data["is_explicit"] == True, "observed_playlist_days"
].mean()

if pd.notna(clean_avg_days) and clean_avg_days > 0 and pd.notna(explicit_avg_days):
    explicit_lifecycle_score = (explicit_avg_days / clean_avg_days) * 100
    explicit_score_text = f"{explicit_lifecycle_score:.1f}"
else:
    explicit_score_text = "N/A"

kpi6.metric(
    "Explicit Content Lifecycle Score",
    explicit_score_text
)


single_avg_days = lifecycle.loc[
    lifecycle["album_type"].astype(str).str.lower() == "single",
    "observed_playlist_days"
].mean()
album_avg_days = lifecycle.loc[
    lifecycle["album_type"].astype(str).str.lower() == "album",
    "observed_playlist_days"
].mean()

if pd.notna(single_avg_days) and pd.notna(album_avg_days) and album_avg_days > 0:
    longevity_ratio = single_avg_days / album_avg_days
    longevity_ratio_text = f"{longevity_ratio:.2f}×"
else:
    longevity_ratio_text = "N/A"

kpi7.metric(
    "Single vs Album Longevity Ratio",
    longevity_ratio_text
)


# --------------------------------------------------
# LIFECYCLE ANALYSIS
# --------------------------------------------------

st.header("2. Song Lifecycle Analysis")

st.write("Song-level lifecycle summary")

display_columns = [
    "song",
    "artist",
    "entry_date",
    "last_seen_date",
    "observed_playlist_days",
    "peak_position",
    "time_to_peak_days",
    "average_popularity"
]

st.dataframe(
    lifecycle[display_columns].sort_values(
        "observed_playlist_days",
        ascending=False
    ),
    use_container_width=True
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.hist(
    lifecycle["observed_playlist_days"],
    bins=20
)

ax.set_title("Distribution of Observed Playlist Days")
ax.set_xlabel("Observed Playlist Days")
ax.set_ylabel("Number of Songs")

st.pyplot(fig)
plt.close(fig)

# Lifecycle stage distribution (uses date/content/album filters, before stage filter).
st.subheader("Lifecycle Stage Distribution")
stage_counts = (
    stage_data["lifecycle_stage"]
    .value_counts()
    .reindex(stage_options, fill_value=0)
)
st.bar_chart(stage_counts)

st.subheader("Song Lifecycle Timeline")
timeline_songs = (
    filtered[["song_id", "song", "artist"]]
    .drop_duplicates()
    .sort_values(["artist", "song"])
)
timeline_labels = {
    row["song_id"]: f'{row["song"]} — {row["artist"]}'
    for _, row in timeline_songs.iterrows()
}
selected_song_id = st.selectbox(
    "Select a song to view its playlist ranking over time",
    options=timeline_songs["song_id"].tolist(),
    format_func=lambda song_id: timeline_labels.get(song_id, song_id)
)
timeline = filtered[filtered["song_id"] == selected_song_id].sort_values("date")
if not timeline.empty:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timeline["date"], timeline["position"], marker="o")
    ax.set_title(f'Playlist Ranking Over Time: {timeline.iloc[0]["song"]}')
    ax.set_xlabel("Date")
    ax.set_ylabel("Playlist Position (1 is best)")
    ax.invert_yaxis()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
   

# --------------------------------------------------
# PLAYLIST ROTATION
# --------------------------------------------------

st.header("3. Playlist Rotation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily Entries and Exits")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        churn_df["date"],
        churn_df["new_entries"],
        label="New Entries"
    )

    ax.plot(
        churn_df["date"],
        churn_df["exits"],
        label="Exits"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Songs")
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Monthly Playlist Rotation")

    monthly = churn_df.set_index("date").resample("MS").agg(
        new_entries=("new_entries", "sum"),
        exits=("exits", "sum")
    )

    st.bar_chart(monthly)

# --------------------------------------------------
# CONTENT MATURITY
# --------------------------------------------------

st.header("4. Explicit vs Clean Songs")

explicit_summary = lifecycle.groupby("is_explicit").agg(
    total_songs=("song_id", "nunique"),
    average_playlist_days=("observed_playlist_days", "mean"),
    average_peak_position=("peak_position", "mean"),
    average_popularity=("average_popularity", "mean"),
    average_time_to_peak=("time_to_peak_days", "mean")
).reset_index()

explicit_summary["content_type"] = (
    explicit_summary["is_explicit"]
    .map({True: "Explicit", False: "Clean"})
)

st.dataframe(
    explicit_summary.drop(columns=["is_explicit"]).round(2),
    use_container_width=True
)



fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    explicit_summary["content_type"],
    explicit_summary["average_playlist_days"]
)

ax.set_title("Average Observed Playlist Days")
ax.set_ylabel("Days")

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# SINGLE VS ALBUM
# --------------------------------------------------

st.header("5. Single vs Album Analysis")

album_summary = lifecycle.groupby("album_type").agg(
    total_songs=("song_id", "nunique"),
    average_playlist_days=("observed_playlist_days", "mean"),
    average_peak_position=("peak_position", "mean"),
    average_popularity=("average_popularity", "mean"),
    average_time_to_peak=("time_to_peak_days", "mean")
).reset_index()

st.dataframe(album_summary.round(2), use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    album_summary["album_type"],
    album_summary["average_playlist_days"]
)

ax.set_title("Single vs Album: Average Observed Days")
ax.set_xlabel("Album Type")
ax.set_ylabel("Observed Playlist Days")

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# POPULARITY AND DURATION
# --------------------------------------------------

st.header("6. Popularity and Duration Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Popularity vs Observed Days")

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(
        lifecycle["average_popularity"],
        lifecycle["observed_playlist_days"],
        alpha=0.6
    )

    ax.set_xlabel("Average Popularity")
    ax.set_ylabel("Observed Playlist Days")

    st.pyplot(fig)
    plt.close(fig)

    popularity_corr = lifecycle[
        ["average_popularity", "observed_playlist_days"]
    ].corr().iloc[0, 1]

    st.write(
        f"Correlation: {popularity_corr:.3f}"
        if pd.notna(popularity_corr)
        else "Correlation: unavailable"
    )

with col2:
    st.subheader("Duration vs Observed Days")

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(
        lifecycle["duration_minutes"],
        lifecycle["observed_playlist_days"],
        alpha=0.6
    )

    ax.set_xlabel("Duration (Minutes)")
    ax.set_ylabel("Observed Playlist Days")

    st.pyplot(fig)
    plt.close(fig)

    duration_corr = lifecycle[
        ["duration_minutes", "observed_playlist_days"]
    ].corr().iloc[0, 1]

    st.write(
        f"Correlation: {duration_corr:.3f}"
        if pd.notna(duration_corr)
        else "Correlation: unavailable"
    )

# --------------------------------------------------
# DOWNLOAD ANALYSIS
# --------------------------------------------------

st.header("7. Download Results")

csv_data = lifecycle.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Song Lifecycle Analysis",
    data=csv_data,
    file_name="spain_top50_song_lifecycle.csv",
    mime="text/csv"
)
