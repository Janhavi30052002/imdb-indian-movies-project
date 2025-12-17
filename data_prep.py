import pandas as pd


def load_raw(path="data/raw/imdb_indian_movies.csv"):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()

    # Basic cleaning
    df = df.dropna(subset=["imdb_rating"])
    df["imdb_rating"] = df["imdb_rating"].astype(float)

    # Fix inconsistent formats
    df["genre"] = df["genre"].fillna("Unknown")
    df["actors"] = df["actors"].fillna("")
    df["director"] = df["director"].fillna("Unknown")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return df


def save_clean(df, path="data/clean/imdb_indian_clean.csv"):
    df.to_csv(path, index=False)
