import pandas as pd


def explode_list(df, col):
    return df.assign(**{col: df[col].str.split(",")}).explode(col)


def actor_experience_feature(df):
    df_exp = explode_list(df, "actors")
    df_exp["actors"] = df_exp["actors"].str.strip()

    df_exp = df_exp.sort_values("year")

    exp_map = df_exp.groupby("actors").cumcount()
    df_exp["actor_experience"] = exp_map.values

    df_exp = df_exp.groupby("movie_id")["actor_experience"].mean()
    return df.join(df_exp, on="movie_id")


def director_experience_feature(df):
    df = df.sort_values("year")
    df["director_experience"] = (
        df.groupby("director").cumcount()
    )
    return df


def build_features(df):
    df = actor_experience_feature(df)
    df = director_experience_feature(df)

    return df
