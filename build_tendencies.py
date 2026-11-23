"""EDGE tendencies builder.

Downloads nflverse play-by-play, FTN charting, snap counts and (when published)
participation/coverage data, then writes small summary tables for game-script
cards:

  data/tendencies/team_offense_{season}.csv      play-calling profile by team (season, last 4, and by week)
  data/tendencies/team_defense_{season}.csv      defensive profile by team
  data/tendencies/player_usage_{season}.csv      target/carry/red-zone shares (season and last 4) + snaps
  data/tendencies/coverage_defense_{season}.csv  man/zone and shell rates by defense (participation seasons only)
  data/tendencies/coverage_receivers_{season}.csv receiver splits vs man and vs zone (participation seasons only)
  data/tendencies/meta.json                      build time, sources, definitions, attribution

Data: nflverse (https://github.com/nflverse/nflverse-data). FTN charting and
participation data are provided by FTN Data via nflverse under CC-BY-SA 4.0;
outputs derived from them keep that attribution.

Definitions are in DEFINITIONS below so every number on a script card can be
traced. Run: python build_tendencies.py [--seasons 2025 2026] [--cache DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
URLS = {
    "pbp": BASE + "/pbp/play_by_play_{season}.csv.gz",
    "ftn": BASE + "/ftn_charting/ftn_charting_{season}.csv",
    "snaps": BASE + "/snap_counts/snap_counts_{season}.csv",
    "part": BASE + "/pbp_participation/pbp_participation_{season}.csv",
}
OUT_DIR = os.path.join("data", "tendencies")
UA = {"User-Agent": "edge-nfl-data/1.0"}

PBP_COLS = [
    "play_id", "game_id", "season", "season_type", "week", "posteam", "defteam",
    "qtr", "down", "ydstogo", "yardline_100", "half_seconds_remaining",
    "game_seconds_remaining", "drive", "score_differential", "wp", "play_type",
    "pass", "rush", "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "sack",
    "shotgun", "no_huddle", "xpass", "pass_oe", "epa", "success", "yards_gained",
    "air_yards", "pass_attempt", "complete_pass", "receiver_player_id",
    "receiver_player_name", "rusher_player_id", "rusher_player_name",
    "receiving_yards", "touchdown", "interception", "two_point_attempt",
]
FTN_COLS = [
    "nflverse_game_id", "nflverse_play_id", "is_play_action", "is_screen_pass",
    "is_rpo", "is_motion", "n_blitzers", "n_pass_rushers", "n_defense_box",
]
PART_COLS = [
    "nflverse_game_id", "play_id", "possession_team", "defense_man_zone_type",
    "defense_coverage_type", "was_pressure", "time_to_throw", "route",
]
SNAP_COLS = ["season", "game_type", "week", "player", "position", "team", "offense_snaps", "offense_pct"]

SINGLE_HIGH = {"COVER_0", "COVER_1", "COVER_3"}
TWO_HIGH = {"COVER_2", "2_MAN", "COVER_4", "COVER_6"}

DEFINITIONS = {
    "offensive play": "pass == 1 or rush == 1 (pass includes sacks and scrambles), excluding kneels, spikes and two-point tries; regular season only",
    "neutral": "win probability between 0.20 and 0.80 and more than 120 seconds left in the half",
    "pass_rate_neutral": "share of neutral offensive plays that are dropbacks",
    "proe_neutral": "mean nflfastR pass_oe on neutral plays, in percentage points (actual pass rate minus expected)",
    "pass_rate_lead8 / trail8": "dropback share when the offense leads / trails by 8+ points",
    "early_down_pass_rate_neutral": "dropback share on neutral 1st and 2nd downs",
    "sec_per_play_neutral": "median seconds between consecutive neutral snaps on the same drive",
    "rz_pass_rate": "dropback share inside the opponent 20; inside10 the same inside the 10",
    "go_rate_4th_short": "share of 4th-and-2-or-less (excluding the last 2 minutes of each half) where the offense ran a play rather than punt or kick",
    "play_action / screen / rpo / motion": "FTN-charted rate; play action and screen per dropback, rpo and motion per offensive play",
    "blitz_rate": "share of dropbacks with 1+ blitzer (FTN n_blitzers > 0)",
    "man_rate": "share of charted dropbacks the defense was in man coverage (participation data; published after each season)",
    "single_high / two_high": "single high = Cover 0/1/3; two high = Cover 2, 2-Man, 4, 6",
    "target_share": "player targets / team pass attempts with a credited receiver",
    "last4": "the team's four most recent games in the data",
    "receiver coverage splits": "targets, target share, yards/target and EPA/target vs man and vs zone; small samples are flagged (fewer than 25 targets) and should be blended toward overall rates before use",
}


# ---------------------------------------------------------------- download
def fetch(kind: str, season: int, cache: str | None) -> pd.DataFrame | None:
    url = URLS[kind].format(season=season)
    usecols = {"pbp": PBP_COLS, "ftn": FTN_COLS, "part": PART_COLS, "snaps": SNAP_COLS}[kind]
    src = url
    if cache:
        local = os.path.join(cache, os.path.basename(url))
        if os.path.exists(local):
            src = local
        else:
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=180) as r, open(local, "wb") as f:
                    f.write(r.read())
                src = local
            except urllib.error.HTTPError as e:
                print(f"  {kind} {season}: not available ({e.code})")
                return None
    try:
        df = pd.read_csv(src, usecols=lambda c: c in set(usecols), low_memory=False,
                         storage_options=None if os.path.exists(src) else UA)
    except urllib.error.HTTPError as e:
        print(f"  {kind} {season}: not available ({e.code})")
        return None
    except Exception as e:  # network or parse failure: keep the rest of the build alive
        print(f"  {kind} {season}: failed to load ({e})")
        return None
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        print(f"  {kind} {season}: missing columns {missing}")
    print(f"  {kind} {season}: {len(df):,} rows")
    return df


# ---------------------------------------------------------------- helpers
def _rate(num: pd.Series, den_mask: pd.Series | None = None) -> float:
    s = num if den_mask is None else num[den_mask]
    s = pd.to_numeric(s, errors="coerce").dropna()
    return round(float(s.mean()), 4) if len(s) else np.nan


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    return s.map({True: 1.0, False: 0.0, "TRUE": 1.0, "FALSE": 0.0, "True": 1.0, "False": 0.0,
                  1: 1.0, 0: 0.0, "1": 1.0, "0": 0.0}).astype(float)


def prep_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp[pbp["season_type"] == "REG"].copy()
    for c in ["pass", "rush", "qb_kneel", "qb_spike", "two_point_attempt", "shotgun", "no_huddle", "sack"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df[((df["pass"] == 1) | (df["rush"] == 1)) & (df["qb_kneel"] == 0) & (df["qb_spike"] == 0)
            & (df.get("two_point_attempt", 0) == 0) & df["posteam"].notna()]
    df["neutral"] = df["wp"].between(0.2, 0.8) & (df["half_seconds_remaining"] > 120)
    df["dropback"] = (df["pass"] == 1).astype(float)
    return df


def add_pace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_id", "play_id"]).copy()
    prev = df.groupby(["game_id", "posteam", "drive"])["game_seconds_remaining"].shift(1)
    df["sec_since_prev"] = prev - df["game_seconds_remaining"]
    df.loc[(df["sec_since_prev"] <= 0) | (df["sec_since_prev"] > 60), "sec_since_prev"] = np.nan
    return df


def last_n_weeks(df: pd.DataFrame, team_col: str, n: int = 4) -> pd.DataFrame:
    games = df[[team_col, "game_id", "week"]].drop_duplicates().sort_values("week")
    keep = games.groupby(team_col).tail(n)[["game_id", team_col]]
    return df.merge(keep, on=["game_id", team_col])


# ---------------------------------------------------------------- offense
def offense_profile(g: pd.DataFrame) -> dict:
    n = g["neutral"]
    sd = g["score_differential"]
    rz, i10 = g["yardline_100"] <= 20, g["yardline_100"] <= 10
    out = {
        "games": g["game_id"].nunique(),
        "plays": len(g),
        "plays_per_game": round(len(g) / max(g["game_id"].nunique(), 1), 1),
        "pass_rate": _rate(g["dropback"]),
        "pass_rate_neutral": _rate(g["dropback"], n),
        "proe_neutral": _rate(g["pass_oe"], n),
        "early_down_pass_rate_neutral": _rate(g["dropback"], n & g["down"].isin([1, 2])),
        "pass_rate_lead8": _rate(g["dropback"], sd >= 8),
        "pass_rate_trail8": _rate(g["dropback"], sd <= -8),
        "sec_per_play_neutral": round(float(g.loc[n, "sec_since_prev"].median()), 1) if n.any() else np.nan,
        "shotgun_rate": _rate(g["shotgun"]),
        "no_huddle_rate": _rate(g["no_huddle"]),
        "rz_pass_rate": _rate(g["dropback"], rz),
        "inside10_pass_rate": _rate(g["dropback"], i10),
        "epa_per_play": _rate(g["epa"]),
        "epa_per_dropback": _rate(g["epa"], g["dropback"] == 1),
        "epa_per_rush": _rate(g["epa"], g["dropback"] == 0),
        "success_rate": _rate(g["success"]),
        "sack_rate": _rate(g["sack"], g["dropback"] == 1),
    }
    for col, name, mask in [("is_play_action", "play_action_rate", g["dropback"] == 1),
                            ("is_screen_pass", "screen_rate", g["dropback"] == 1),
                            ("is_rpo", "rpo_rate", None), ("is_motion", "motion_rate", None)]:
        out[name] = _rate(g[col], mask) if col in g else np.nan
    if "n_blitzers" in g:
        out["blitz_faced_rate"] = _rate((g["n_blitzers"] > 0).astype(float).where(g["n_blitzers"].notna()), g["dropback"] == 1)
    return out


def fourth_down(all_plays: pd.DataFrame) -> pd.DataFrame:
    """Go rate on 4th-and-short from the raw pbp (punts and kicks are not offensive plays)."""
    d = all_plays[(all_plays["season_type"] == "REG") & (all_plays["down"] == 4) & (all_plays["ydstogo"] <= 2)
                  & (all_plays["half_seconds_remaining"] > 120)
                  & all_plays["play_type"].isin(["pass", "run", "punt", "field_goal"])].copy()
    d["go"] = d["play_type"].isin(["pass", "run"]).astype(float)
    return d.groupby("posteam").agg(go_att_4th_short=("go", "sum"), go_rate_4th_short=("go", "mean")).round(3).reset_index()


def team_offense(df: pd.DataFrame, raw: pd.DataFrame, playcallers: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for team, g in df.groupby("posteam"):
        rows.append({"team": team, "split": "season", "week": "", **offense_profile(g)})
    for team, g in last_n_weeks(df, "posteam").groupby("posteam"):
        rows.append({"team": team, "split": "last4", "week": "", **offense_profile(g)})
    for (team, wk), g in df.groupby(["posteam", "week"]):
        rows.append({"team": team, "split": "week", "week": int(wk), **offense_profile(g)})
    out = pd.DataFrame(rows)
    fd = fourth_down(raw).rename(columns={"posteam": "team"}).assign(split="season")
    out = out.merge(fd, on=["team", "split"], how="left")  # season-level only; too few attempts per week
    if playcallers is not None and len(playcallers):
        out["play_caller"] = ""
        for _, pc in playcallers.iterrows():
            m = (out["team"] == pc["team"]) & (out["split"] == "week")
            m &= pd.to_numeric(out["week"], errors="coerce").between(int(pc["from_week"]), int(pc["to_week"]))
            out.loc[m, "play_caller"] = pc["play_caller"]
            sub = df[(df["posteam"] == pc["team"]) & df["week"].between(int(pc["from_week"]), int(pc["to_week"]))]
            if len(sub):
                out = pd.concat([out, pd.DataFrame([{"team": pc["team"], "split": "play_caller", "week": "",
                                                     "play_caller": pc["play_caller"], **offense_profile(sub)}])],
                                ignore_index=True)
    return out


# ---------------------------------------------------------------- defense
def defense_profile(g: pd.DataFrame) -> dict:
    db = g["dropback"] == 1
    out = {
        "games": g["game_id"].nunique(),
        "epa_allowed_per_play": _rate(g["epa"]),
        "epa_allowed_per_dropback": _rate(g["epa"], db),
        "epa_allowed_per_rush": _rate(g["epa"], ~db),
        "success_allowed": _rate(g["success"]),
        "sack_rate": _rate(g["sack"], db),
        "explosive_pass_allowed_rate": _rate((g["yards_gained"] >= 20).astype(float), db),
        "explosive_rush_allowed_rate": _rate((g["yards_gained"] >= 10).astype(float), ~db),
        "opp_pass_rate_neutral": _rate(g["dropback"], g["neutral"]),
        "opp_proe_neutral": _rate(g["pass_oe"], g["neutral"]),
    }
    if "n_blitzers" in g:
        nb = g["n_blitzers"]
        out["blitz_rate"] = _rate((nb > 0).astype(float).where(nb.notna()), db)
        out["avg_pass_rushers"] = _rate(g["n_pass_rushers"], db)
        out["avg_box"] = _rate(g["n_defense_box"], ~db)
    return out


def team_defense(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"team": t, "split": "season", **defense_profile(g)} for t, g in df.groupby("defteam")]
    rows += [{"team": t, "split": "last4", **defense_profile(g)} for t, g in last_n_weeks(df, "defteam").groupby("defteam")]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- players
def usage(g: pd.DataFrame, team_g: pd.DataFrame) -> pd.DataFrame:
    tgt = g[g["receiver_player_id"].notna() & (g["dropback"] == 1)]
    team_tgts = tgt.groupby("posteam").size()
    team_air = tgt.groupby("posteam")["air_yards"].sum()
    team_rz_tgts = tgt[tgt["yardline_100"] <= 20].groupby("posteam").size()
    rec = tgt.groupby(["posteam", "receiver_player_id"]).agg(
        player=("receiver_player_name", "last"), targets=("play_id", "size"),
        receptions=("complete_pass", "sum"), rec_yards=("receiving_yards", "sum"),
        air_yards=("air_yards", "sum"),
        rz_targets=("yardline_100", lambda s: int((s <= 20).sum())),
        epa_per_target=("epa", "mean")).reset_index().rename(columns={"receiver_player_id": "player_id"})
    rec["target_share"] = (rec["targets"] / rec["posteam"].map(team_tgts)).round(3)
    rec["air_yards_share"] = (rec["air_yards"] / rec["posteam"].map(team_air)).round(3)
    rec["rz_target_share"] = (rec["rz_targets"] / rec["posteam"].map(team_rz_tgts)).round(3)

    ru = g[(g["dropback"] == 0) & g["rusher_player_id"].notna()]
    team_car = ru.groupby("posteam").size()
    team_i10 = ru[ru["yardline_100"] <= 10].groupby("posteam").size()
    car = ru.groupby(["posteam", "rusher_player_id"]).agg(
        player=("rusher_player_name", "last"), carries=("play_id", "size"),
        rush_yards=("yards_gained", "sum"),
        inside10_carries=("yardline_100", lambda s: int((s <= 10).sum())),
        epa_per_rush=("epa", "mean")).reset_index().rename(columns={"rusher_player_id": "player_id"})
    car["carry_share"] = (car["carries"] / car["posteam"].map(team_car)).round(3)
    car["inside10_carry_share"] = (car["inside10_carries"] / car["posteam"].map(team_i10)).round(3)

    out = rec.merge(car, on=["posteam", "player_id"], how="outer", suffixes=("", "_ru"))
    out["player"] = out["player"].fillna(out.pop("player_ru"))
    out["games"] = out["posteam"].map(team_g)
    return out.rename(columns={"posteam": "team"})


def player_usage(df: pd.DataFrame, snaps: pd.DataFrame | None) -> pd.DataFrame:
    games = df.groupby("posteam")["game_id"].nunique()
    l4 = last_n_weeks(df, "posteam")
    s = usage(df, games).assign(split="season")
    l = usage(l4, l4.groupby("posteam")["game_id"].nunique()).assign(split="last4")
    out = pd.concat([s, l], ignore_index=True)
    if snaps is not None and len(snaps):
        sn = snaps[snaps["game_type"] == "REG"].copy()
        sn["offense_pct"] = pd.to_numeric(sn["offense_pct"], errors="coerce")
        last4_weeks = sn.sort_values("week").groupby("team")["week"].apply(lambda w: set(sorted(w.unique())[-4:]))
        sn["in_last4"] = [wk in last4_weeks.get(t, set()) for t, wk in zip(sn["team"], sn["week"])]
        agg = lambda d: d.groupby(["team", "player"]).agg(snap_pct=("offense_pct", "mean"), snap_games=("week", "nunique")).round(3).reset_index()
        snap_tbl = pd.concat([agg(sn).assign(split="season"), agg(sn[sn["in_last4"]]).assign(split="last4")])
        # nflverse pbp uses short names ("J.Jefferson"); snap counts use full names. Match on team + last name + first initial.
        def key(name: str) -> str:
            name = str(name).replace(".", " ").split()
            return (name[0][0] + "|" + name[-1]).lower() if len(name) >= 2 else str(name).lower()
        snap_tbl["k"] = snap_tbl["player"].map(key)
        out["k"] = out["player"].map(key)
        snap_tbl = snap_tbl.drop_duplicates(["team", "split", "k"])
        out = out.merge(snap_tbl.drop(columns="player"), on=["team", "split", "k"], how="left").drop(columns="k")
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    return out.sort_values(["team", "split", "targets"], ascending=[True, True, False])


# ---------------------------------------------------------------- coverage (participation seasons)
def coverage_tables(df: pd.DataFrame, part: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = part.rename(columns={"nflverse_game_id": "game_id"})
    m = df.merge(p, on=["game_id", "play_id"], how="inner")
    m = m[m["dropback"] == 1]
    m["mz"] = m["defense_man_zone_type"].astype(str).str.upper()
    m["is_man"] = m["mz"].str.contains("MAN").astype(float).where(m["mz"].str.contains("MAN|ZONE"))
    cov = m["defense_coverage_type"].astype(str).str.upper()
    m["single_high"] = cov.isin(SINGLE_HIGH).astype(float).where(cov.isin(SINGLE_HIGH | TWO_HIGH))
    m["pressured"] = _bool(m["was_pressure"]) if "was_pressure" in m else np.nan

    def drow(g):
        cov_counts = g["defense_coverage_type"].astype(str).str.upper().value_counts(normalize=True)
        r = {"dropbacks": len(g), "man_rate": _rate(g["is_man"]), "single_high_rate": _rate(g["single_high"]),
             "pressure_rate": _rate(g["pressured"]), "time_to_throw": _rate(g["time_to_throw"]),
             "man_rate_3rd_down": _rate(g["is_man"], g["down"] == 3),
             "epa_allowed_vs_man": _rate(g["epa"], g["is_man"] == 1),
             "epa_allowed_vs_zone": _rate(g["epa"], g["is_man"] == 0)}
        for c in ["COVER_0", "COVER_1", "COVER_2", "2_MAN", "COVER_3", "COVER_4", "COVER_6"]:
            r[c.lower() + "_rate"] = round(float(cov_counts.get(c, 0.0)), 3)
        return r
    dcov = pd.DataFrame([{"team": t, **drow(g)} for t, g in m.groupby("defteam")])

    t = m[m["receiver_player_id"].notna() & m["is_man"].notna()].copy()
    t["cov"] = np.where(t["is_man"] == 1, "man", "zone")
    team_t = t.groupby(["posteam", "cov"]).size().rename("team_targets")
    r = t.groupby(["posteam", "receiver_player_id", "cov"]).agg(
        player=("receiver_player_name", "last"), targets=("play_id", "size"),
        receptions=("complete_pass", "sum"), yards=("receiving_yards", "sum"), epa=("epa", "sum")).reset_index()
    r = r.join(team_t, on=["posteam", "cov"])
    r["target_share"] = (r["targets"] / r["team_targets"]).round(3)
    r["yards_per_target"] = (r["yards"] / r["targets"]).round(2)
    r["epa_per_target"] = (r["epa"] / r["targets"]).round(3)
    wide = r.pivot_table(index=["posteam", "receiver_player_id", "player"], columns="cov",
                         values=["targets", "target_share", "yards_per_target", "epa_per_target"]).reset_index()
    wide.columns = ["_".join([c for c in col if c]).strip("_") for col in wide.columns]
    wide = wide.rename(columns={"posteam": "team", "receiver_player_id": "player_id"})
    for c in ["targets_man", "targets_zone"]:
        if c not in wide:
            wide[c] = 0
    wide["small_sample_man"] = wide["targets_man"].fillna(0) < 25
    wide["small_sample_zone"] = wide["targets_zone"].fillna(0) < 25
    if {"target_share_man", "target_share_zone"} <= set(wide.columns):
        wide["share_shift_man_minus_zone"] = (wide["target_share_man"] - wide["target_share_zone"]).round(3)
    return dcov, wide.sort_values(["team", "targets_man"], ascending=[True, False])


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    this_year = dt.date.today().year if dt.date.today().month >= 8 else dt.date.today().year - 1
    ap.add_argument("--seasons", nargs="+", type=int, default=[this_year - 1, this_year])
    ap.add_argument("--cache", default=None, help="directory for downloaded files (optional)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    pc_path = os.path.join("data", "playcallers.csv")
    playcallers = pd.read_csv(pc_path) if os.path.exists(pc_path) else None
    meta = {"built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "seasons": {}, "definitions": DEFINITIONS,
            "attribution": "Data: nflverse. FTN charting and participation data by FTN Data via nflverse, CC-BY-SA 4.0."}

    for season in args.seasons:
        print(f"Season {season}")
        raw = fetch("pbp", season, args.cache)
        if raw is None:
            continue
        df = add_pace(prep_pbp(raw))
        ftn = fetch("ftn", season, args.cache)
        if ftn is not None:
            ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
            for c in ["is_play_action", "is_screen_pass", "is_rpo", "is_motion"]:
                if c in ftn:
                    ftn[c] = _bool(ftn[c])
            df = df.merge(ftn, on=["game_id", "play_id"], how="left")
        snaps = fetch("snaps", season, args.cache)
        part = fetch("part", season, args.cache)

        pcs = playcallers[playcallers["season"] == season] if playcallers is not None and "season" in playcallers else None
        files = {
            f"team_offense_{season}.csv": team_offense(df, raw, pcs),
            f"team_defense_{season}.csv": team_defense(df),
            f"player_usage_{season}.csv": player_usage(df, snaps),
        }
        if part is not None and len(part):
            dcov, rcov = coverage_tables(df, part)
            files[f"coverage_defense_{season}.csv"] = dcov
            files[f"coverage_receivers_{season}.csv"] = rcov
        for name, table in files.items():
            table.to_csv(os.path.join(OUT_DIR, name), index=False)
            print(f"  wrote {name}: {len(table):,} rows")
        meta["seasons"][str(season)] = {
            "max_week": int(df["week"].max()), "plays": int(len(df)),
            "ftn": ftn is not None, "snaps": snaps is not None, "participation": part is not None,
            "files": sorted(files),
        }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return 0 if meta["seasons"] else 1


if __name__ == "__main__":
    sys.exit(main())
