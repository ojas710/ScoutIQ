# Data

ScoutIQ uses the **Transfermarkt football dataset** curated by *dcaribou*
(`dcaribou/transfermarkt-datasets`, published on Kaggle as `davidcariboo/player-scores`).

## Obtain it
1. Download the dataset from Kaggle (free account) or build it from the GitHub project.
2. Place these CSV files in `data/raw/` (a filename prefix is tolerated, e.g. `123_games.csv`), or point
   `SCOUTIQ_RAW_DIR` at their folder:
   `players.csv`, `player_valuations.csv`, `appearances.csv`, `games.csv`, `clubs.csv`, `competitions.csv`.

## Licensing — please verify
The dataset is listed on Kaggle with a permissive (CC0) licence, but the underlying values originate from
Transfermarkt, whose terms of use apply to the source data. Confirm the current licence on the dataset page before
redistributing anything. `data/raw/` is git-ignored; this repo does not redistribute the raw files.
`data/processed/` holds derived player-season aggregates (~6 MB) so the app runs without the raw data; remove
it from version control if you are not comfortable redistributing derived data.

## Schema notes (verified on the file version used for the reported results)
* `player_valuations`: ~656k dated valuations (2000–2026), 14 domestic leagues have matching appearance data.
* `appearances`: per-match goals, assists, cards, minutes (no shots, xG, xA, passes, starts).
* Current-snapshot columns (e.g. `players.market_value_in_eur`, caps, contract expiry, club totals) are **not used**
  because they would leak future information.
