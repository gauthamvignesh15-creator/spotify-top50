"""Data loading, validation, cleaning and feature engineering.
Usage:  python clean.py Atlantic_United_States.csv
Creates: cleaned_playlist.csv (one row per song per day), song_summary.csv (one row per song)
"""
import sys
import pandas as pd
import numpy as np


def load_clean(path):
    df = pd.read_csv(path)
    report = {"raw_rows": len(df)}

    # dates are dd-mm-yyyy
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

    # text standardisation
    for c in ["song", "artist"]:
        df[c] = df[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["album_type"] = df["album_type"].str.strip().str.lower()
    df["is_explicit"] = df["is_explicit"].astype(bool)

    # validation
    report["bad_rank"] = int((~df["position"].between(1, 50)).sum())
    df = df[df["position"].between(1, 50)]
    report["missing_values"] = int(df.drop(columns="album_cover_url").isna().sum().sum())

    # a date loaded twice (01-03-2025 has 100 rows): keep only the first 50-row snapshot
    report["extra_snapshot_rows"] = int((df.groupby("date").cumcount() >= 50).sum())
    df = df[df.groupby("date").cumcount() < 50]

    # remaining duplicates: same song twice on same date
    report["exact_duplicate_rows"] = int(df.duplicated().sum())
    report["duplicate_song_date"] = int(df.duplicated(["date", "song", "artist"]).sum())
    df = df.sort_values(["date", "position"]).drop_duplicates(["date", "song", "artist"], keep="first")

    # popularity 0 is an API glitch (a song cannot be at 0 yet in the Top 50): treat as missing
    report["popularity_zero_as_missing"] = int((df["popularity"] == 0).sum())
    df.loc[df["popularity"] == 0, "popularity"] = np.nan

    # derived columns
    df["song_id"] = df["song"] + " — " + df["artist"]
    df["duration_min"] = df["duration_ms"] / 60000
    df["main_artist"] = (df["artist"].str.replace("Tyler, The Creator", "Tyler The Creator", regex=False)
                         .str.split(r" & |, ", regex=True).str[0]
                         .str.replace("Tyler The Creator", "Tyler, The Creator", regex=False))
    df["is_collab"] = df["artist"].str.contains(r" & |, ", regex=True) & (df["artist"] != "Tyler, The Creator")
    df["tier"] = pd.cut(df["position"], [0, 10, 20, 50], labels=["Top 10", "Top 11-20", "Top 21-50"])

    df = df.sort_values(["song_id", "date"]).reset_index(drop=True)
    g = df.groupby("song_id")
    # rank change vs previous appearance (positive = moved up), only for consecutive days
    gap = g["date"].diff().dt.days
    df["rank_change"] = (g["position"].shift(1) - df["position"]).where(gap == 1)
    # Popularity Trend Score: 7-day rolling mean of popularity per song
    df["pop_trend"] = g["popularity"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df = df.sort_values(["date", "position"]).reset_index(drop=True)
    report["clean_rows"] = len(df)
    report["days"] = int(df["date"].nunique())
    return df, report


def song_summary(df):
    s = df.groupby("song_id").agg(
        song=("song", "first"), artist=("artist", "first"), main_artist=("main_artist", "first"),
        days_on_chart=("date", "nunique"),
        avg_rank=("position", "mean"), best_rank=("position", "min"),
        rank_volatility=("position", "std"),
        avg_popularity=("popularity", "mean"), pop_volatility=("popularity", "std"),
        first_date=("date", "min"), last_date=("date", "max"),
        is_explicit=("is_explicit", "first"), album_type=("album_type", "first"),
        total_tracks=("total_tracks", "first"), duration_min=("duration_min", "first"),
        max_rise=("rank_change", "max"), max_fall=("rank_change", "min"),
    ).reset_index()
    s["rank_volatility"] = s["rank_volatility"].fillna(0)
    s["span_days"] = (s["last_date"] - s["first_date"]).dt.days + 1
    return s


def artist_summary(df):
    total_slots = len(df)
    a = df.groupby("main_artist").agg(
        unique_songs=("song_id", "nunique"), total_days=("date", "size"),
        avg_rank=("position", "mean"), best_rank=("position", "min"),
        avg_popularity=("popularity", "mean")).reset_index()
    a["dominance_index"] = a["total_days"] / total_slots * 100   # % of all chart slots
    return a.sort_values("dominance_index", ascending=False)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "Atlantic_United_States.csv"
    df, rep = load_clean(src)
    df.to_csv("cleaned_playlist.csv", index=False)
    song_summary(df).to_csv("song_summary.csv", index=False)
    artist_summary(df).to_csv("artist_summary.csv", index=False)
    print("Validation report:")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("Saved cleaned_playlist.csv, song_summary.csv, artist_summary.csv")
