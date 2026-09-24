#!/usr/bin/env python3

import csv
import io
import pathlib
import urllib.request

SEASONS = range(2022, 2027)

BASE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)

OUTPUT_DIR = pathlib.Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

COMMON_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "position_group",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
]

OFFENSE_COLUMNS = COMMON_COLUMNS + [
    "completions",
    "attempts",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_air_yards",
    "passing_yards_after_catch",
    "passing_first_downs",
    "passing_epa",
    "passing_cpoe",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    "rushing_epa",
    "receptions",
    "targets",
    "receiving_yards",
    "receiving_tds",
    "receiving_air_yards",
    "receiving_yards_after_catch",
    "receiving_first_downs",
    "receiving_epa",
    "target_share",
    "air_yards_share",
    "wopr",
    "pacr",
    "racr",
]

DEFENSE_COLUMNS = COMMON_COLUMNS + [
    "def_tackles_solo",
    "def_tackles_with_assist",
    "def_tackle_assists",
    "def_tackles_for_loss",
    "def_tackles_for_loss_yards",
    "def_fumbles_forced",
    "def_sacks",
    "def_sack_yards",
    "def_qb_hits",
    "def_interceptions",
    "def_interception_yards",
    "def_pass_defended",
    "def_tds",
    "def_fumbles",
    "def_safeties",
]


def download_season(season):
    url = BASE_URL.format(season=season)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "edge-nfl-data/1.0"},
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8-sig")


def save_filtered(rows, columns, filename, filter_function):
    path = OUTPUT_DIR / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            if filter_function(row):
                writer.writerow(
                    {column: row.get(column, "") for column in columns}
                )


for season in SEASONS:

    print(f"Downloading nflverse {season} player data...")

    csv_text = download_season(season)

    rows = list(
        csv.DictReader(
            io.StringIO(csv_text)
        )
    )

    save_filtered(
        rows,
        OFFENSE_COLUMNS,
        f"offense_{season}.csv",
        lambda row: (
            row.get("position_group") in {"QB", "RB", "WR", "TE"}
            or row.get("position") == "FB"
        ),
    )

    save_filtered(
        rows,
        DEFENSE_COLUMNS,
        f"defense_{season}.csv",
        lambda row: (
            row.get("position_group") in {"DL", "LB", "DB"}
        ),
    )

print("Edge nflverse data build complete.")
