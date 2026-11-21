"""Create a readable top-20 TrueSkill-ranked team map-pool report."""

import os

import duckdb
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(REPO_ROOT, "vallu_scraper", "data", "parquet")
FEATURES_DIR = os.path.join(REPO_ROOT, "vallu_scraper", "data", "features")
OUTPUT_DIR = os.path.join(REPO_ROOT, "vallu_scraper", "data", "analysis")
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "team_map_pool_top20_trueskill.html")
TOP_TEAM_COUNT = 20
ROSTER_TRUESKILL_FILE = os.path.join(
    FEATURES_DIR,
    "roster_trueskill",
    "features_roster_trueskill_10.000_6.000_0.250.parquet",
)


def build_team_map_summary() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]
]:
    """Return map details, team overview, picks, and data coverage counts."""
    matches_path = os.path.join(PARQUET_DIR, "matches.parquet").replace("'", "''")
    maps_path = os.path.join(PARQUET_DIR, "maps.parquet").replace("'", "''")
    vetos_path = os.path.join(PARQUET_DIR, "vetos.parquet").replace("'", "''")

    con = duckdb.connect()
    detail = con.execute(
        f"""
        WITH matches AS (
            SELECT
                hltv_match_id,
                NULLIF(TRIM(team1_name), '') AS team1_name,
                NULLIF(TRIM(team2_name), '') AS team2_name
            FROM read_parquet('{matches_path}')
        ),
        team_matches AS (
            SELECT team1_name AS team, hltv_match_id FROM matches
            UNION ALL
            SELECT team2_name AS team, hltv_match_id FROM matches
        ),
        team_totals AS (
            SELECT team, COUNT(DISTINCT hltv_match_id) AS team_matches
            FROM team_matches
            WHERE team IS NOT NULL
            GROUP BY team
        ),
        map_plays_raw AS (
            SELECT
                m.team1_name AS team,
                LOWER(TRIM(mp.map_name)) AS map_name,
                COUNT(*) AS maps_played
            FROM matches m
            JOIN read_parquet('{maps_path}') mp USING (hltv_match_id)
            WHERE m.team1_name IS NOT NULL
              AND NULLIF(TRIM(mp.map_name), '') IS NOT NULL
            GROUP BY m.team1_name, LOWER(TRIM(mp.map_name))
            UNION ALL
            SELECT
                m.team2_name AS team,
                LOWER(TRIM(mp.map_name)) AS map_name,
                COUNT(*) AS maps_played
            FROM matches m
            JOIN read_parquet('{maps_path}') mp USING (hltv_match_id)
            WHERE m.team2_name IS NOT NULL
              AND NULLIF(TRIM(mp.map_name), '') IS NOT NULL
            GROUP BY m.team2_name, LOWER(TRIM(mp.map_name))
        ),
        map_plays AS (
            SELECT team, map_name, SUM(maps_played) AS maps_played
            FROM map_plays_raw
            GROUP BY team, map_name
        ),
        bans_against AS (
            SELECT
                CASE LOWER(TRIM(v.team))
                    WHEN 'team1' THEN m.team2_name
                    WHEN 'team2' THEN m.team1_name
                END AS team,
                LOWER(TRIM(v.map)) AS map_name,
                COUNT(*) AS bans_against,
                COUNT(DISTINCT v.hltv_match_id) AS series_with_ban
            FROM read_parquet('{vetos_path}') v
            JOIN matches m USING (hltv_match_id)
            WHERE LOWER(TRIM(v.action)) = 'removed'
              AND LOWER(TRIM(v.team)) IN ('team1', 'team2')
              AND NULLIF(TRIM(v.map), '') IS NOT NULL
            GROUP BY 1, 2
        ),
        ban_totals AS (
            SELECT team, SUM(bans_against) AS total_bans_against
            FROM bans_against
            WHERE team IS NOT NULL
            GROUP BY team
        )
        SELECT
            COALESCE(p.team, b.team) AS team,
            COALESCE(p.map_name, b.map_name) AS map_name,
            COALESCE(p.maps_played, 0) AS maps_played,
            COALESCE(b.bans_against, 0) AS bans_against,
            COALESCE(b.series_with_ban, 0) AS series_with_ban,
            tt.team_matches,
            COALESCE(bt.total_bans_against, 0) AS total_bans_against
        FROM map_plays p
        FULL OUTER JOIN bans_against b
          ON p.team = b.team AND p.map_name = b.map_name
        LEFT JOIN team_totals tt ON tt.team = COALESCE(p.team, b.team)
        LEFT JOIN ban_totals bt ON bt.team = COALESCE(p.team, b.team)
        ORDER BY team, map_name
        """
    ).df()
    picks = con.execute(
        f"""
        WITH matches AS (
            SELECT
                hltv_match_id,
                NULLIF(TRIM(team1_name), '') AS team1_name,
                NULLIF(TRIM(team2_name), '') AS team2_name
            FROM read_parquet('{matches_path}')
        )
        SELECT
            CASE LOWER(TRIM(v.team))
                WHEN 'team1' THEN m.team1_name
                WHEN 'team2' THEN m.team2_name
            END AS team,
            LOWER(TRIM(v.map)) AS map_name,
            COUNT(*) AS pick_count,
            COUNT(*) FILTER (
                WHERE TRY_CAST(REGEXP_EXTRACT(TRIM(mp.score), '(\\d+):(\\d+)', 1) AS INTEGER) IS NOT NULL
                  AND TRY_CAST(REGEXP_EXTRACT(TRIM(mp.score), '(\\d+):(\\d+)', 2) AS INTEGER) IS NOT NULL
            ) AS maps_played_after_pick,
            SUM(CASE
                WHEN LOWER(TRIM(v.team)) = 'team1'
                 AND TRY_CAST(REGEXP_EXTRACT(TRIM(mp.score), '(\\d+):(\\d+)', 1) AS INTEGER)
                     > TRY_CAST(REGEXP_EXTRACT(TRIM(mp.score), '(\\d+):(\\d+)', 2) AS INTEGER)
                    THEN 1
                WHEN LOWER(TRIM(v.team)) = 'team2'
                 AND TRY_CAST(REGEXP_EXTRACT(TRIM(mp.score), '(\\d+):(\\d+)', 2) AS INTEGER)
                     > TRY_CAST(REGEXP_EXTRACT(TRIM(mp.score), '(\\d+):(\\d+)', 1) AS INTEGER)
                    THEN 1
                ELSE 0
            END) AS map_wins_when_picked
        FROM read_parquet('{vetos_path}') v
        JOIN matches m USING (hltv_match_id)
        LEFT JOIN read_parquet('{maps_path}') mp
          ON mp.hltv_match_id = v.hltv_match_id
         AND LOWER(TRIM(mp.map_name)) = LOWER(TRIM(v.map))
        WHERE LOWER(TRIM(v.action)) = 'picked'
          AND LOWER(TRIM(v.team)) IN ('team1', 'team2')
          AND NULLIF(TRIM(v.map), '') IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    veto_matches = con.execute(
        f"""
        WITH matches AS (
            SELECT
                hltv_match_id,
                NULLIF(TRIM(team1_name), '') AS team1_name,
                NULLIF(TRIM(team2_name), '') AS team2_name
            FROM read_parquet('{matches_path}')
        ),
        team_veto_matches AS (
            SELECT m.team1_name AS team, v.hltv_match_id
            FROM read_parquet('{vetos_path}') v
            JOIN matches m USING (hltv_match_id)
            UNION ALL
            SELECT m.team2_name AS team, v.hltv_match_id
            FROM read_parquet('{vetos_path}') v
            JOIN matches m USING (hltv_match_id)
        )
        SELECT team, COUNT(DISTINCT hltv_match_id) AS team_veto_matches
        FROM team_veto_matches
        WHERE team IS NOT NULL
        GROUP BY team
        """
    ).df()
    coverage = con.execute(
        f"""
        SELECT
            (SELECT COUNT(DISTINCT hltv_match_id)
             FROM read_parquet('{matches_path}')) AS total_matches,
            (SELECT COUNT(DISTINCT hltv_match_id)
             FROM read_parquet('{maps_path}')) AS matches_with_maps,
            (SELECT COUNT(DISTINCT hltv_match_id)
             FROM read_parquet('{vetos_path}')) AS matches_with_vetos
        """
    ).df().iloc[0].to_dict()
    con.close()

    detail["map_name"] = detail["map_name"].str.title()
    detail["maps_played"] = detail["maps_played"].astype(int)
    detail["bans_against"] = detail["bans_against"].astype(int)
    detail["series_with_ban"] = detail["series_with_ban"].astype(int)
    detail["team_matches"] = detail["team_matches"].astype(int)
    detail["total_bans_against"] = detail["total_bans_against"].astype(int)
    detail["map_play_share_pct"] = (
        detail["maps_played"]
        .div(detail.groupby("team")["maps_played"].transform("sum"))
        .mul(100)
    )
    detail["ban_share_pct"] = (
        detail["bans_against"]
        .div(detail["total_bans_against"].where(detail["total_bans_against"] > 0))
        .mul(100)
    )
    detail = detail.merge(veto_matches, on="team", how="left")
    detail["team_veto_matches"] = detail["team_veto_matches"].fillna(0).astype(int)
    detail["ban_target_rate_pct"] = (
        detail["series_with_ban"]
        .div(detail["team_veto_matches"].where(detail["team_veto_matches"] > 0))
        .mul(100)
    )
    detail["map_play_rank"] = detail.groupby("team")["maps_played"].rank(
        method="min", ascending=False
    ).astype(int)
    detail["ban_rank"] = detail.groupby("team")["bans_against"].rank(
        method="min", ascending=False
    ).where(detail["bans_against"] > 0).astype("Int64")

    def format_top(group: pd.DataFrame, count_col: str, share_col: str) -> tuple[str, int, float]:
        top = group[group[count_col] == group[count_col].max()]
        top = top[top[count_col] > 0]
        if top.empty:
            return "", 0, 0.0
        labels = [
            f"{row.map_name} ({int(getattr(row, count_col))})"
            for row in top.sort_values("map_name").itertuples(index=False)
        ]
        return ", ".join(labels), int(top[count_col].iloc[0]), float(top[share_col].iloc[0])

    overview_rows = []
    for team, group in detail.groupby("team", sort=True):
        played, played_count, played_share = format_top(
            group, "maps_played", "map_play_share_pct"
        )
        banned, banned_count, banned_share = format_top(
            group, "bans_against", "ban_share_pct"
        )
        overview_rows.append(
            {
                "team": team,
                "team_matches": int(group["team_matches"].iloc[0]),
                "total_maps_played": int(group["maps_played"].sum()),
                "most_played_map(s)": played,
                "most_played_count_each": played_count,
                "most_played_share_pct_each": played_share,
                "total_opponent_bans_recorded": int(
                    group["total_bans_against"].iloc[0]
                ),
                "most_banned_map_against_team(s)": banned,
                "most_banned_count_each": banned_count,
                "most_banned_share_pct_each": banned_share,
            }
        )
    overview = pd.DataFrame(overview_rows)

    picks["map_name"] = picks["map_name"].str.title()
    picks["pick_count"] = picks["pick_count"].astype(int)
    picks["maps_played_after_pick"] = picks["maps_played_after_pick"].astype(int)
    picks["map_wins_when_picked"] = picks["map_wins_when_picked"].astype(int)
    picks["team_pick_total"] = picks.groupby("team")["pick_count"].transform("sum")
    picks["pick_rate_pct"] = (
        picks["pick_count"].div(picks["team_pick_total"]).mul(100)
    )
    picks["pick_win_rate_pct"] = (
        picks["map_wins_when_picked"]
        .div(picks["maps_played_after_pick"].where(picks["maps_played_after_pick"] > 0))
        .mul(100)
    )

    return (
        detail,
        overview,
        picks,
        {key: int(value) for key, value in coverage.items()},
    )


def build_last_team_ratings() -> pd.DataFrame:
    """Get each exact team name's latest pre-match conservative roster rating."""
    matches_path = os.path.join(PARQUET_DIR, "matches.parquet").replace("'", "''")
    ratings_path = ROSTER_TRUESKILL_FILE.replace("'", "''")
    con = duckdb.connect()
    ratings = con.execute(
        f"""
        WITH match_ratings AS (
            SELECT
                NULLIF(TRIM(m.team1_name), '') AS team,
                TRY_CAST(m.date AS TIMESTAMP) AS last_match_date,
                m.hltv_match_id AS last_match_id,
                r.team1_roster_trueskill_conservative AS last_pre_match_roster_trueskill_conservative
            FROM read_parquet('{matches_path}') m
            JOIN read_parquet('{ratings_path}') r USING (hltv_match_id)
            UNION ALL
            SELECT
                NULLIF(TRIM(m.team2_name), '') AS team,
                TRY_CAST(m.date AS TIMESTAMP) AS last_match_date,
                m.hltv_match_id AS last_match_id,
                r.team2_roster_trueskill_conservative AS last_pre_match_roster_trueskill_conservative
            FROM read_parquet('{matches_path}') m
            JOIN read_parquet('{ratings_path}') r USING (hltv_match_id)
        )
        SELECT
            team,
            last_match_date,
            last_match_id,
            last_pre_match_roster_trueskill_conservative
        FROM match_ratings
        WHERE team IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY team
            ORDER BY last_match_date DESC NULLS LAST, last_match_id DESC
        ) = 1
        """
    ).df()
    con.close()
    return ratings


def main() -> None:
    detail, overview, picks, coverage = build_team_map_summary()
    last_ratings = build_last_team_ratings()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ranked = overview.merge(last_ratings, on="team", how="left")
    ranked = ranked.sort_values(
        ["last_pre_match_roster_trueskill_conservative", "team"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(
        0,
        "trueskill_rank",
        ranked["last_pre_match_roster_trueskill_conservative"]
        .rank(method="min", ascending=False)
        .astype("Int64"),
    )
    top_teams = ranked.head(TOP_TEAM_COUNT)
    overview_columns = [
        "trueskill_rank",
        "team",
        "team_matches",
        "total_maps_played",
        "most_played_map(s)",
        "most_played_share_pct_each",
        "most_banned_map_against_team(s)",
        "most_banned_share_pct_each",
        "last_pre_match_roster_trueskill_conservative",
    ]
    overview_table = top_teams[overview_columns].rename(
        columns={
            "trueskill_rank": "Rank",
            "team": "Team",
            "team_matches": "Matches",
            "total_maps_played": "Maps played",
            "most_played_map(s)": "Most played map(s)",
            "most_played_share_pct_each": "Most played share (%)",
            "most_banned_map_against_team(s)": "Most targeted ban(s)",
            "most_banned_share_pct_each": "Share of opponent bans (%)",
            "last_pre_match_roster_trueskill_conservative": "Latest TrueSkill (pre-match)",
        }
    ).copy()
    overview_table["Most played share (%)"] = overview_table[
        "Most played share (%)"
    ].map(lambda value: f"{value:.1f}%")
    overview_table["Share of opponent bans (%)"] = overview_table[
        "Share of opponent bans (%)"
    ].map(lambda value: f"{value:.1f}%")
    overview_table["Latest TrueSkill (pre-match)"] = overview_table[
        "Latest TrueSkill (pre-match)"
    ].map(lambda value: f"{value:.2f}")

    top_team_names = set(top_teams["team"])
    team_ranks = top_teams.set_index("team")["trueskill_rank"]
    top_picks = picks[picks["team"].isin(top_team_names)].copy()
    top_picks["trueskill_rank"] = top_picks["team"].map(team_ranks)
    top_picks = (
        top_picks.sort_values(
            ["trueskill_rank", "pick_count", "map_name"],
            ascending=[True, False, True],
        )
        .groupby("team", sort=False)
        .head(5)
        .copy()
    )
    top_picks["pick_rank"] = top_picks.groupby("team")["pick_count"].rank(
        method="first", ascending=False
    ).astype(int)
    picks_table = top_picks[
        [
            "trueskill_rank",
            "team",
            "pick_rank",
            "map_name",
            "pick_count",
            "pick_rate_pct",
            "map_wins_when_picked",
            "maps_played_after_pick",
            "pick_win_rate_pct",
        ]
    ].rename(
        columns={
            "trueskill_rank": "Team rank",
            "team": "Team",
            "pick_rank": "Pick rank",
            "map_name": "Picked map",
            "pick_count": "Times picked",
            "pick_rate_pct": "Share of team picks",
            "map_wins_when_picked": "Map wins after pick",
            "maps_played_after_pick": "Maps played after pick",
            "pick_win_rate_pct": "Win rate when picked",
        }
    ).copy()
    picks_table["Share of team picks"] = picks_table["Share of team picks"].map(
        lambda value: f"{value:.1f}%"
    )
    picks_table["Win rate when picked"] = picks_table[
        "Win rate when picked"
    ].map(lambda value: f"{value:.1f}%" if pd.notna(value) else "N/A")

    top_bans = detail[
        detail["team"].isin(top_team_names) & (detail["bans_against"] > 0)
    ].copy()
    top_bans["trueskill_rank"] = top_bans["team"].map(team_ranks)
    top_bans = (
        top_bans.sort_values(
            ["trueskill_rank", "bans_against", "map_name"],
            ascending=[True, False, True],
        )
        .groupby("team", sort=False)
        .head(3)
        .copy()
    )
    top_bans["ban_rank"] = top_bans.groupby("team")["bans_against"].rank(
        method="first", ascending=False
    ).astype(int)
    bans_table = top_bans[
        [
            "trueskill_rank",
            "team",
            "ban_rank",
            "map_name",
            "bans_against",
            "series_with_ban",
            "team_veto_matches",
            "ban_target_rate_pct",
            "ban_share_pct",
        ]
    ].rename(
        columns={
            "trueskill_rank": "Team rank",
            "team": "Team",
            "ban_rank": "Ban rank",
            "map_name": "Map opponents banned",
            "bans_against": "Times opponents banned it",
            "series_with_ban": "Series targeted",
            "team_veto_matches": "Team series with veto data",
            "ban_target_rate_pct": "Target rate (% of veto series)",
            "ban_share_pct": "Share of opponent bans",
        }
    ).copy()
    bans_table["Target rate (% of veto series)"] = bans_table[
        "Target rate (% of veto series)"
    ].map(lambda value: f"{value:.1f}%")
    bans_table["Share of opponent bans"] = bans_table[
        "Share of opponent bans"
    ].map(lambda value: f"{value:.1f}%")

    veto_coverage_pct = (
        100 * coverage["matches_with_vetos"] / coverage["total_matches"]
        if coverage["total_matches"]
        else 0.0
    )
    html_content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Top 20 CS2 Team Map Pool</title>
<style>
body {{ font-family: Arial, sans-serif; color: #202a35; margin: 24px auto; max-width: 1500px; padding: 0 20px; }}
h1, h2 {{ color: #152b43; }}
.note {{ color: #526170; line-height: 1.5; }}
nav {{ margin: 18px 0 28px; display: flex; gap: 18px; flex-wrap: wrap; }}
nav a {{ color: #0759a5; font-weight: 600; }}
.table-wrap {{ overflow-x: auto; margin: 14px 0 34px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th {{ background: #193b5a; color: white; text-align: left; position: sticky; top: 0; }}
th, td {{ border: 1px solid #d5dde5; padding: 8px 10px; white-space: nowrap; }}
tbody tr:nth-child(even) {{ background: #f1f5f8; }}
tbody tr:hover {{ background: #e2effa; }}
</style>
</head>
<body>
<h1>Top 20 team map pool</h1>
<p class="note">Teams are ranked by latest pre-match conservative roster TrueSkill
from the 10.000 / 6.000 / 0.250 feature set. Map veto data is available for
{coverage['matches_with_vetos']:,} of {coverage['total_matches']:,} matches
({veto_coverage_pct:.1f}%). Team names are grouped exactly as recorded.</p>
<nav>
<a href="#overview">Overview</a>
<a href="#picks">Picked maps and win rates</a>
<a href="#bans">Maps opponents target</a>
</nav>
<h2 id="overview">Team overview</h2>
<div class="table-wrap">{overview_table.to_html(index=False, border=0, classes="data-table", escape=True, na_rep="N/A")}</div>
<h2 id="picks">Top five veto-picked maps per team</h2>
<p class="note">Pick share is the map's percentage of that team's recorded veto picks.
Win rate is wins divided by map results available after that team picked the map.</p>
<div class="table-wrap">{picks_table.to_html(index=False, border=0, classes="data-table", escape=True, na_rep="N/A")}</div>
<h2 id="bans">Top three maps opponents banned against each team</h2>
<p class="note">Only opponent <em>removed</em> actions count here; bans made by the team itself are excluded.
Target rate is the percentage of that team's veto-recorded series where an opponent banned the map.
The final percentage is its share of all recorded opponent bans against that team.</p>
<div class="table-wrap">{bans_table.to_html(index=False, border=0, classes="data-table", escape=True, na_rep="N/A")}</div>
</body>
</html>
"""
    with open(OUTPUT_HTML, "w", encoding="utf-8") as output_file:
        output_file.write(html_content)

    print(
        f"Matches: {coverage['total_matches']}; "
        f"with maps: {coverage['matches_with_maps']}; "
        f"with veto data: {coverage['matches_with_vetos']} "
        f"({veto_coverage_pct:.1f}%)."
    )
    print(f"Saved top-{TOP_TEAM_COUNT} readable report -> {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
