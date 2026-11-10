"""Streamlit dashboard.  Run:  streamlit run app.py
Needs Atlantic_United_States.csv in the same folder.  pip install streamlit plotly pandas
"""
import pandas as pd
import plotly.express as px
import streamlit as st
from clean import load_clean

st.set_page_config(page_title="US Top 50 Playlist Analytics", layout="wide")
BLUE, ORANGE = "#1f6feb", "#e8590c"


@st.cache_data
def get_data():
    df, _ = load_clean("Atlantic_United_States.csv")
    return df


df = get_data()
st.title("United States Top 50 — Playlist Performance & Popularity Trends")
st.caption("Atlantic Recording Corporation · historical playlist analytics")

# ---------------- sidebar filters ----------------
st.sidebar.header("Filters")
dmin, dmax = df.date.min().date(), df.date.max().date()
dr = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(dr, tuple) and len(dr) == 2:
    d0, d1 = dr
else:
    d0 = d1 = dr[0] if isinstance(dr, (tuple, list)) else dr
rank_lo, rank_hi = st.sidebar.slider("Rank range", 1, 50, (1, 50))
types = sorted(df.album_type.unique())
sel_types = st.sidebar.multiselect("Album type", types, default=types)
artists = st.sidebar.multiselect("Artist(s)", sorted(df.main_artist.unique()))
song_pool = df[df.main_artist.isin(artists)] if artists else df
songs = st.sidebar.multiselect("Song(s)", sorted(song_pool.song_id.unique()))

f = df[(df.date.dt.date >= d0) & (df.date.dt.date <= d1)
       & df.position.between(rank_lo, rank_hi) & df.album_type.isin(sel_types)]
if artists:
    f = f[f.main_artist.isin(artists)]
if songs:
    f = f[f.song_id.isin(songs)]

if f.empty:
    st.warning("No data for these filters. Widen the date range, rank range or album type.")
    st.stop()

# ---------------- KPIs ----------------
per_song_days = f.groupby("song_id").size()
total_slots = len(df)
k = st.columns(6)
k[0].metric("Avg days on chart / song", f"{per_song_days.mean():.1f}")
k[1].metric("Average rank", f"{f.position.mean():.1f}")
k[2].metric("Rank volatility (std)", f"{f.position.std():.1f}")
k[3].metric("Avg popularity", f"{f.popularity.mean():.1f}")
k[4].metric("Top artist dominance", f"{f.main_artist.value_counts().iloc[0] / len(f) * 100:.1f}%",
            help="Share of the filtered chart slots held by the #1 artist")
k[5].metric("Explicit share", f"{f.is_explicit.mean() * 100:.1f}%")

t1, t2, t3, t4, t5, t6 = st.tabs(["Timeline explorer", "Song rank trends", "Artist leaderboard",
                                   "Popularity vs rank", "Explicit vs non-explicit", "Data"])

# ---------------- 1 timeline ----------------
with t1:
    day = st.date_input("Pick a day to view the full Top 50", f.date.max().date(), min_value=dmin, max_value=dmax, key="day")
    snap = df[df.date.dt.date == day].sort_values("position")
    st.dataframe(snap[["position", "song", "artist", "popularity", "album_type", "is_explicit"]],
                 hide_index=True, use_container_width=True, height=380)
    c1, c2 = st.columns(2)
    daily = f.groupby("date").agg(avg_pop=("popularity", "mean"), explicit=("is_explicit", "mean")).reset_index()
    c1.plotly_chart(px.line(daily, x="date", y="avg_pop", title="Average popularity per day",
                            color_discrete_sequence=[BLUE]), use_container_width=True)
    entries = df.groupby("song_id").date.min().rename("first").reset_index()
    entries = entries[(entries["first"].dt.date >= d0) & (entries["first"].dt.date <= d1)]
    weekly = entries.set_index("first").resample("W").size().reset_index(name="new songs")
    c2.plotly_chart(px.bar(weekly, x="first", y="new songs", title="New song entries per week",
                           color_discrete_sequence=[ORANGE]).update_xaxes(title=""), use_container_width=True)

# ---------------- 2 song trends ----------------
with t2:
    pool = f.song_id.value_counts()
    default = pool.index[:3].tolist()
    chosen = st.multiselect("Songs to compare", pool.index.tolist(), default=default)
    if chosen:
        sd = f[f.song_id.isin(chosen)]
        fig = px.line(sd, x="date", y="position", color="song_id", title="Daily rank (1 = top)")
        fig.update_yaxes(autorange="reversed", range=[50, 1])
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.line(sd, x="date", y="pop_trend", color="song_id", title="Popularity trend (7-day rolling average)")
        st.plotly_chart(fig2, use_container_width=True)
    S = f.groupby("song_id").agg(days_on_chart=("date", "nunique"), avg_rank=("position", "mean"),
                                 best_rank=("position", "min"), rank_volatility=("position", "std"),
                                 avg_popularity=("popularity", "mean")).round(2).reset_index()
    st.subheader("Song performance table")
    st.dataframe(S.sort_values("days_on_chart", ascending=False), hide_index=True, use_container_width=True)
    st.plotly_chart(px.scatter(S, x="best_rank", y="days_on_chart", hover_name="song_id",
                               title="Peak rank vs longevity", color_discrete_sequence=[BLUE])
                    .update_xaxes(autorange="reversed"), use_container_width=True)

# ---------------- 3 artists ----------------
with t3:
    n = st.slider("Artists to show", 5, 25, 15)
    A = f.groupby("main_artist").agg(unique_songs=("song_id", "nunique"), total_days=("date", "size"),
                                     avg_rank=("position", "mean"), avg_popularity=("popularity", "mean")).reset_index()
    A["dominance_index_%"] = (A.total_days / len(f) * 100).round(2)
    A = A.sort_values("total_days", ascending=False)
    top = A.head(n)
    st.plotly_chart(px.bar(top.iloc[::-1], x="total_days", y="main_artist", orientation="h",
                           title="Total chart days (song-days)", color_discrete_sequence=[BLUE])
                    .update_yaxes(title=""), use_container_width=True)
    st.dataframe(A.round(2), hide_index=True, use_container_width=True)
    top5 = A.head(5).main_artist
    m = f[f.main_artist.isin(top5)].copy()
    m["month"] = m.date.dt.to_period("M").dt.to_timestamp()
    mm = m.groupby(["month", "main_artist"]).size().reset_index(name="days")
    st.plotly_chart(px.line(mm, x="month", y="days", color="main_artist", markers=True,
                            title="Top 5 artists: chart presence by month"), use_container_width=True)

# ---------------- 4 popularity vs rank ----------------
with t4:
    p = f.dropna(subset=["popularity"])
    samp = p.sample(min(len(p), 6000), random_state=1)
    st.plotly_chart(px.scatter(samp, x="position", y="popularity", opacity=.35, hover_name="song_id",
                               title="Popularity vs rank", color_discrete_sequence=[BLUE])
                    .update_xaxes(title="Rank"), use_container_width=True)
    st.caption(f"Correlation (rank, popularity): {p.position.corr(p.popularity):.3f}. "
               "Negative means better-ranked songs tend to be more popular.")
    st.plotly_chart(px.box(p, x="tier", y="popularity", title="Popularity by tier (Top 10 / 11-20 / 21-50)",
                           color_discrete_sequence=[BLUE]).update_xaxes(title=""), use_container_width=True)

# ---------------- 5 explicit ----------------
with t5:
    f2 = f.assign(content=f.is_explicit.map({True: "Explicit", False: "Non-explicit"}))
    sg = f2.groupby(["content", "song_id"]).agg(days=("date", "nunique"), avg_rank=("position", "mean"),
                                                avg_pop=("popularity", "mean")).reset_index()
    cmp_ = sg.groupby("content").agg(songs=("song_id", "count"), avg_days=("days", "mean"),
                                     avg_rank=("avg_rank", "mean"), avg_popularity=("avg_pop", "mean")).round(2)
    st.dataframe(cmp_, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.pie(f2, names="content", title="Share of chart slots", hole=.5,
                           color_discrete_sequence=[ORANGE, BLUE]), use_container_width=True)
    mon = f2.assign(month=f2.date.dt.to_period("M").dt.to_timestamp()).groupby("month").is_explicit.mean().mul(100).reset_index()
    c2.plotly_chart(px.line(mon, x="month", y="is_explicit", markers=True, title="Explicit share by month (%)",
                            color_discrete_sequence=[ORANGE]), use_container_width=True)
    st.plotly_chart(px.box(f2, x="content", y="position", title="Rank distribution", color="content",
                           color_discrete_sequence=[ORANGE, BLUE]).update_yaxes(autorange="reversed"),
                    use_container_width=True)

# ---------------- 6 data ----------------
with t6:
    st.dataframe(f.drop(columns=["album_cover_url"]), hide_index=True, use_container_width=True)
    st.download_button("Download filtered data (CSV)", f.to_csv(index=False), "filtered_playlist.csv", "text/csv")
