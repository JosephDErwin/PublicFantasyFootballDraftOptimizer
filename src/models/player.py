from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil import parser
from scipy.stats import lognorm, gamma, norm

from espn_api_mod.football.constant import PRO_TEAM_MAP, PLAYER_STATS_MAP
from src.config import N_SIMS, SEASON
from src.constants import GAMES_IN_SEASON

LAGS = [1, 3, 10, 20]


def convert_stats(stats):
    """Converts stat keys from ESPN's integer codes to human-readable strings."""
    new_dict = {}
    for i in stats:
        if int(i) in PLAYER_STATS_MAP:
            new_dict[PLAYER_STATS_MAP[int(i)]] = stats.get(i, 0)
    return new_dict


def find_sleeper(espn_player, players):
    """Matches an ESPN player object to a Sleeper player object."""
    if 'D/ST' in espn_player.pro_position:
        return None

    # Attempt matching by various attributes in order of reliability
    candidates = [x for x in players.values() if x.get('espn_id', 0) == espn_player.id]
    if len(candidates) == 1:
        return candidates[0]

    candidates = [x for x in players.values() if x.get('full_name', 0) == espn_player.name]
    if len(candidates) == 1:
        return candidates[0]

    candidates = [x for x in players.values() if x.get('position', 0) == espn_player.pro_position]
    for name_part in espn_player.name.split(' '):
        candidates = [x for x in candidates if name_part in x.get('full_name', '')]
        if len(candidates) == 1:
            return candidates[0]
    return None


class Player():
    """
    Represents a single fantasy football player, holding all their static info,
    historical stats, and future projections for IN-SEASON weekly analysis.
    """

    def __init__(self, fantasy_player, league, extra_players, opponent_schedule, game_lookup, bios, draft_data,
                 game_logs, season, byes):
        # Core Attributes
        self.simulations = pd.DataFrame()
        self.rolling_cols = None
        self.name = fantasy_player.name
        self.id = fantasy_player.playerId
        self.positions = fantasy_player.eligibleSlots
        self.pro_position = fantasy_player.position
        self.pro_team = {'OAK': 'LV'}.get(fantasy_player.proTeam, fantasy_player.proTeam)
        self.bye_week = byes.get(self.pro_team, None)
        self.manager = None
        self.agg_level = ['season', 'week_num']
        self.season = season
        self.lags = LAGS
        self.current_age = None

        # Injury & Status
        self.injured = fantasy_player.injured
        self.status = fantasy_player.injuryStatus

        # Projections & VORP - Initialized here, set later
        self.ave_projected_ppg = 0
        self.ave_play_prob = 0
        self.projected_dists = {}
        self.projected_ppg = 0
        self.vorp = 0

        # Raw Game Data
        self.game_logs = pd.DataFrame({})
        self.game_stats = pd.DataFrame({})
        self.schedule = opponent_schedule.get(self.pro_team, {})

        # Draft & Roster Info
        self.draft_round = league.draft_dict.get(self.id, None)
        self.locked = extra_players.get(self.id, {}).get('rosterLocked', False)
        if fantasy_player.standardRank and fantasy_player.pprRank:
            self.adp = (fantasy_player.standardRank + fantasy_player.pprRank) / 2
        else:
            self.adp = None

        # Bio & Draft Info
        self.draft_info = self.get_draft_data(draft_data)
        bio = find_sleeper(self, bios)
        if bio:
            self.birthday = parser.parse(bio['birth_date']) if bio.get('birth_date') else None
            self.current_age = (datetime.now() - self.birthday).days / 365.25 if self.birthday else None
            self.height = float(bio['height']) if bio.get('height') else None
            self.weight = float(bio['weight']) if bio.get('weight') else None
            self.first_season = league.year - bio.get('years_exp', 0)
            self.college = bio.get('college')
            self.current_experience = self.season - self.first_season
        else:
            self.birthday, self.height, self.weight, self.first_season  = None, None, None, None
            self.college, self.current_experience = None, None

        # Ownership & Value
        extra_info = extra_players.get(self.id)
        if extra_info:
            self.rostered = extra_info['player']['ownership']['percentOwned']
            self.roster_change = extra_info['player']['ownership']['percentChange']
        else:
            self.rostered, self.roster_change = None, None

        # Process historical game logs
        self.set_stats(game_logs, game_lookup)

    def get_draft_data(self, draft_data):
        """Finds draft data for the player by ID or by fuzzy name matching."""
        data = draft_data.get(self.id)
        if data:
            return data

        # Fallback to name matching if ID lookup fails
        name_parts = self.name.split(' ')
        candidate_list = list(draft_data.values())
        for part in name_parts:
            subset = [p for p in candidate_list if part.lower() in p.name.lower()]
            if len(subset) == 1:
                return subset[0]
            elif len(subset) > 1:
                candidate_list = subset
            else:
                return None  # No match found
        return candidate_list[0] if len(candidate_list) > 1 else None

    def create_game_info(self, info, matchup):
        """Builds a dictionary of player and game context for a single game log."""
        opponent_id = matchup['awayProTeamId'] if info['proTeamId'] == matchup['homeProTeamId'] else matchup[
            'homeProTeamId']

        age = (info['date'] - self.birthday).days / 365.25 if self.birthday else None
        experience = info['seasonId'] - self.first_season if self.first_season else None

        game_info = {
            'name': self.name,
            'season': info['seasonId'],
            'team': PRO_TEAM_MAP.get(info['proTeamId'], info['proTeamId']),
            'opponent': PRO_TEAM_MAP.get(opponent_id),
            'player_id': self.id,
            'week_num': info['scoringPeriodId'],
            'position': self.pro_position,
            'height': self.height,
            'weight': self.weight,
            'age': age,
            'experience': experience,
            'college': self.college,
        }

        if self.draft_info:
            game_info['draft_pick'] = self.draft_info.overall
            game_info['draft_grade'] = self.draft_info.pre_draft_grade
            game_info['pre_draft_ranking'] = self.draft_info.pre_draft_ranking
            game_info['college_conference'] = self.draft_info.college_conference

        return game_info

    def set_stats(self, game_logs, game_lookup):
        """Processes raw game logs into structured DataFrames."""
        game_logs_unfiltered = game_logs.get(self.id, [])
        if not game_logs_unfiltered: return

        for i in game_logs_unfiltered:
            i['date'] = datetime(i['seasonId'], 9, 7) + timedelta(i['scoringPeriodId'] * 7)

        logs = list(filter(lambda x: x['statSplitTypeId'] == 1, game_logs_unfiltered))
        logs.sort(key=lambda x: x['id'])

        stats_logs, game_info_logs = {}, {}
        for i in logs:
            if not i['stats']: continue
            matchup = game_lookup.get(i['externalId'])
            if not matchup: continue

            stats = convert_stats(i['stats'])
            game_info = self.create_game_info(i, matchup)
            game_info['applied_total'] = i.get('appliedTotal', 0)
            game_info['date'] = i['date']

            stats_logs[i['id']] = stats
            game_info_logs[i['id']] = game_info

        logs_df = pd.DataFrame(stats_logs).fillna(0).T
        stats_df = pd.DataFrame(game_info_logs).T
        df = pd.concat([stats_df, logs_df], axis=1)

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

        self.game_logs = df
        self.game_stats = pd.DataFrame(stats_logs).T.fillna(0)

    def games_played_mat(self, point_categories, current_week):
        """Creates a matrix of historical games for model training with enhanced features."""
        if self.game_logs.empty: return pd.DataFrame([])

        games = self.game_logs
        if games.empty: return pd.DataFrame([])

        # Find the minimum and maximum values for season and week
        min_season = games['season'].min()
        max_season = games['season'].max()

        # Use these ranges to generate a complete set of historical indexers
        all_seasons_range = range(min_season, max_season + 1)
        all_weeks_range = range(1, 18)

        # The full index includes all combinations in the historical range
        full_index = pd.MultiIndex.from_product(
            [all_seasons_range, all_weeks_range],
            names=['season', 'week_num']
        )

        # Filter out future seasons and weeks that are beyond the last observed historical game.
        # This assumes that the last row of your sorted games DataFrame represents the most recent game.
        last_game = games.sort_values(by=['season', 'week_num']).iloc[-1]
        last_season = last_game['season']
        last_week = last_game['week_num']

        # Retain only the indexers up to and including the last observed game
        full_index = full_index[~((full_index.get_level_values('season') > last_season) |
                                  ((full_index.get_level_values('season') == last_season) &
                                   (full_index.get_level_values('week_num') > last_week)))]

        # Set the index and reindex
        games = games.set_index(['season', 'week_num']).sort_index()
        full_games = games.reindex(full_index)

        full_games['gamesPlayed'] = full_games['gamesPlayed'].fillna(0)
        df = full_games.groupby(level=0).ffill().reset_index()

        df['id'] = self.id
        df['height'] = self.height
        df['weight'] = self.weight
        df['position'] = self.pro_position
        df['college'] = self.college

        if self.draft_info:
            df['draft_pick'] = self.draft_info.overall
            df['draft_grade'] = self.draft_info.pre_draft_grade
            df['pre_draft_ranking'] = self.draft_info.pre_draft_ranking
            df['college_conference'] = self.draft_info.college_conference

        if self.birthday and 'date' in df.columns:
            df['age'] = (df['date'] - self.birthday).dt.days / 365.25
        else:
            df['age'] = None

        desired_cols = list(PLAYER_STATS_MAP.values())
        self.rolling_cols = list(set(desired_cols) & set(df.columns))

        # Enhanced feature engineering with multiple statistical measures
        lagged_cols = [df]
        for lag in self.lags:
            # Original rolling means
            stat_cols_mean = df[self.rolling_cols].shift(1).rolling(lag, min_periods=1).mean()
            stat_cols_mean.columns = stat_cols_mean.columns + f'_{lag}game'
            lagged_cols.append(stat_cols_mean)
            
            # Add rolling standard deviation for volatility
            stat_cols_std = df[self.rolling_cols].shift(1).rolling(lag, min_periods=1).std()
            stat_cols_std.columns = stat_cols_std.columns + f'_{lag}game_std'
            lagged_cols.append(stat_cols_std)
            
            # Add rolling momentum (recent trend)
            if lag >= 5:  # Only for longer windows
                recent_half = lag // 2
                stat_cols_recent = df[self.rolling_cols].shift(1).rolling(recent_half, min_periods=1).mean()
                stat_cols_older = df[self.rolling_cols].shift(recent_half + 1).rolling(recent_half, min_periods=1).mean()
                stat_cols_momentum = stat_cols_recent - stat_cols_older
                stat_cols_momentum.columns = stat_cols_momentum.columns + f'_{lag}game_momentum'
                lagged_cols.append(stat_cols_momentum)

        df = pd.concat(lagged_cols, axis=1)
        
        # Add position-specific efficiency metrics
        df = self._add_position_specific_features(df)
        df['adp'] = df['season'].map(lambda x: self.adp.get(x))
        
        return df

    def _add_position_specific_features(self, df):
        """Adds position-specific efficiency and performance metrics."""
        if self.pro_position == 'QB':
            # QB efficiency features
            for lag in self.lags:
                attempts_col = f'passingAttempts_{lag}game'
                completions_col = f'passingCompletions_{lag}game'
                yards_col = f'passingYards_{lag}game'
                
                if attempts_col in df.columns and completions_col in df.columns:
                    df[f'completion_rate_{lag}game'] = (df[completions_col] / df[attempts_col]).fillna(0)
                
                if attempts_col in df.columns and yards_col in df.columns:
                    df[f'yards_per_attempt_{lag}game'] = (df[yards_col] / df[attempts_col]).fillna(0)
                    
        elif self.pro_position == 'RB':
            # RB efficiency features
            for lag in self.lags:
                attempts_col = f'rushingAttempts_{lag}game'
                yards_col = f'rushingYards_{lag}game'
                targets_col = f'receivingTargets_{lag}game'
                receptions_col = f'receivingReceptions_{lag}game'
                
                if attempts_col in df.columns and yards_col in df.columns:
                    df[f'yards_per_carry_{lag}game'] = (df[yards_col] / df[attempts_col]).fillna(0)
                    
                if targets_col in df.columns and receptions_col in df.columns:
                    df[f'target_share_{lag}game'] = (df[receptions_col] / df[targets_col]).fillna(0)
                    
        elif self.pro_position in ['WR', 'TE']:
            # WR/TE receiving efficiency features
            for lag in self.lags:
                targets_col = f'receivingTargets_{lag}game'
                receptions_col = f'receivingReceptions_{lag}game'
                yards_col = f'receivingYards_{lag}game'
                
                if targets_col in df.columns and receptions_col in df.columns:
                    df[f'catch_rate_{lag}game'] = (df[receptions_col] / df[targets_col]).fillna(0)
                
                if receptions_col in df.columns and yards_col in df.columns:
                    df[f'yards_per_reception_{lag}game'] = (df[yards_col] / df[receptions_col]).fillna(0)
                    
        # Add consistency metrics for all positions
        for lag in self.lags:
            points_col = f'applied_total_{lag}game'
            points_std_col = f'applied_total_{lag}game_std'
            
            if points_col in df.columns and points_std_col in df.columns:
                # Coefficient of variation as consistency metric
                df[f'consistency_{lag}game'] = (df[points_std_col] / df[points_col]).fillna(0)
                # Floor/ceiling metrics  
                df[f'upside_{lag}game'] = df[points_col] + df[points_std_col]
                df[f'floor_{lag}game'] = df[points_col] - df[points_std_col]
        
        return df
        
    def get_rolling_stats(self):
        """Calculates enhanced rolling statistics for the player's game logs."""
        if self.game_logs.empty or self.rolling_cols is None:
            return {}

        resultant_dict = {}
        for lag in self.lags:
            # Original rolling means
            rolling = self.game_logs[self.rolling_cols].rolling(lag, min_periods=1).mean()
            rolling.columns = rolling.columns + f'_{lag}game'
            resultant_dict.update(rolling.iloc[-1].to_dict())
            
            # Add rolling standard deviation for volatility
            rolling_std = self.game_logs[self.rolling_cols].rolling(lag, min_periods=1).std()
            rolling_std.columns = rolling_std.columns + f'_{lag}game_std'
            resultant_dict.update(rolling_std.iloc[-1].to_dict())
            
            # Add rolling momentum (recent trend) for longer windows
            if lag >= 5:
                recent_half = lag // 2
                recent_avg = self.game_logs[self.rolling_cols].rolling(recent_half, min_periods=1).mean().iloc[-1]
                older_avg = self.game_logs[self.rolling_cols].shift(recent_half).rolling(recent_half, min_periods=1).mean().iloc[-1]
                momentum = recent_avg - older_avg
                momentum.index = momentum.index + f'_{lag}game_momentum'
                resultant_dict.update(momentum.to_dict())
        
        # Add position-specific efficiency metrics using current game logs
        if not self.game_logs.empty:
            pos_features = self._calculate_position_efficiency_features()
            resultant_dict.update(pos_features)

        return resultant_dict
        
    def _calculate_position_efficiency_features(self):
        """Calculate position-specific efficiency features from recent game logs."""
        features = {}
        
        if self.pro_position == 'QB':
            for lag in self.lags:
                recent_games = self.game_logs.tail(lag)
                if not recent_games.empty and 'passingAttempts' in recent_games.columns:
                    total_attempts = recent_games['passingAttempts'].sum()
                    total_completions = recent_games.get('passingCompletions', pd.Series([0])).sum()
                    total_yards = recent_games.get('passingYards', pd.Series([0])).sum()
                    
                    if total_attempts > 0:
                        features[f'completion_rate_{lag}game'] = total_completions / total_attempts
                        features[f'yards_per_attempt_{lag}game'] = total_yards / total_attempts
                        
        elif self.pro_position == 'RB':
            for lag in self.lags:
                recent_games = self.game_logs.tail(lag)
                if not recent_games.empty and 'rushingAttempts' in recent_games.columns:
                    total_carries = recent_games['rushingAttempts'].sum()
                    total_rush_yards = recent_games.get('rushingYards', pd.Series([0])).sum()
                    total_targets = recent_games.get('receivingTargets', pd.Series([0])).sum()
                    total_receptions = recent_games.get('receivingReceptions', pd.Series([0])).sum()
                    
                    if total_carries > 0:
                        features[f'yards_per_carry_{lag}game'] = total_rush_yards / total_carries
                    if total_targets > 0:
                        features[f'target_share_{lag}game'] = total_receptions / total_targets
                        
        elif self.pro_position in ['WR', 'TE']:
            for lag in self.lags:
                recent_games = self.game_logs.tail(lag)
                if not recent_games.empty and 'receivingTargets' in recent_games.columns:
                    total_targets = recent_games['receivingTargets'].sum()
                    total_receptions = recent_games.get('receivingReceptions', pd.Series([0])).sum()
                    total_yards = recent_games.get('receivingYards', pd.Series([0])).sum()
                    
                    if total_targets > 0:
                        features[f'catch_rate_{lag}game'] = total_receptions / total_targets
                    if total_receptions > 0:
                        features[f'yards_per_reception_{lag}game'] = total_yards / total_receptions
        
        # Add consistency metrics for all positions
        for lag in self.lags:
            recent_games = self.game_logs.tail(lag)
            if not recent_games.empty and 'applied_total' in recent_games.columns:
                points = recent_games['applied_total']
                if len(points) > 1:
                    mean_points = points.mean()
                    std_points = points.std()
                    if mean_points > 0:
                        features[f'consistency_{lag}game'] = std_points / mean_points
                        features[f'upside_{lag}game'] = mean_points + std_points
                        features[f'floor_{lag}game'] = mean_points - std_points
        
        return features

    def get_future_matchups(self):
        """Generates a list of future games for this player for prediction."""
        rolling_stats = self.get_rolling_stats()

        X = []
        for week, info in self.schedule.items():
            game_date = datetime.fromtimestamp(info[0]['date'] / 1000.0)
            if game_date <= datetime.now(): continue

            summary = {'seasonId': SEASON,
                       'proTeamId': self.pro_team,
                       'scoringPeriodId': week,
                       'date': game_date}

            game_dict = self.create_game_info(summary, info[0])
            game_dict.update(rolling_stats)
            game_dict['adp'] = self.adp.get(SEASON)

            if self.injured and game_date <= (datetime.now() + timedelta(weeks=2)):
                game_dict['gamesPlayed'] = 0
            else:
                game_dict['gamesPlayed'] = 1

            X.append(game_dict)
        return X

    def set_projections(self, df, dist_type, rvs):
        """Sets future game projections from model output in a vectorized way."""
        dist_map = {'gamma': gamma, 'lognormal': lognorm, 'normal': norm}
        dist = dist_map.get(dist_type.lower(), norm)

        # Set the MultiIndex on the input DataFrame for alignment
        df_indexed = df.set_index(['week_num'])

        self.projected_dists = df_indexed['params'].map(lambda x: dist(**x))
        self.projected_ppg = self.projected_dists.map(lambda x: x.mean())

        # Align the projected distributions and rvs by their shared index
        # Stack the distributions so they can be accessed by (player_id, season, week_num)
        dist_stack = self.projected_dists.to_frame(name='dist').reset_index().set_index(['week_num'])

        # Merge the distributions with the rvs DataFrame to align them
        aligned_df = rvs.merge(dist_stack, left_index=True, right_index=True)

        # Vectorized PPF application using a list comprehension
        # This is the most efficient way to apply a method across a pandas Series
        sim_data = np.array([
            row['dist'].ppf(row.drop('dist').astype(float))
            for _, row in aligned_df.iterrows()
        ])

        # Re-shape the results and create the final DataFrame
        # Transpose the sim_data and create a DataFrame with a MultiIndex
        self.simulations = pd.DataFrame(sim_data.T * self.play_sims.values, columns=aligned_df.index)

        self.ave_projected_ppg = self.projected_ppg.mean()

    def set_injury_probs(self, df):
        """Sets future injury probabilities from model output."""
        self.expected_games_played = df.set_index(self.agg_level)['injury_rate']
        self.ave_play_prob = self.expected_games_played.mean()

        self.play_sims = pd.DataFrame(np.random.binomial(n=1, p=self.expected_games_played, size=(N_SIMS, len(self.expected_games_played))),
                                      columns=self.expected_games_played.index)

        self.play_sims.iloc[:, 0] = 1

    def set_vorp(self, replacement_levels):
        """Calculates Value Over Replacement Player (VORP)."""
        if self.ave_projected_ppg is None or self.ave_play_prob is None:
            self.vorp = 0
            return

        vorps = []
        for position in self.positions:
            if position in ['BE', 'IR'] or position not in replacement_levels:
                continue
            vorps.append(self.ave_projected_ppg - replacement_levels[position])

        if vorps:
            # VORP PPG * Expected Games Played
            self.vorp = max(vorps) * (GAMES_IN_SEASON * self.ave_play_prob)
        else:
            self.vorp = 0

    def add_adp(self, adp_data):
        """Adds ADP data to the player object."""

        self.adp = {}
        for season, season_adp in adp_data.items():
            if season < (self.first_season or season): continue

            data = season_adp[['Player Name', 'AVG']]
            subset = data[season_adp['Player Name'] == self.name]

            if subset.shape[0] == 1:
                self.adp[season] = subset['AVG'].values[0]
                continue

            for part in self.name.split(' '):
                subset = data[data['Player Name'].str.contains(part, na=False)]

                if subset.shape[0] == 1:
                    self.adp[season] = subset['AVG'].values[0]
                elif subset.shape[0] < 1:
                    continue

        if not self.game_logs.empty:
            self.game_logs['adp'] = self.game_logs['season'].map(self.adp)

    def __str__(self):
        return self.name


class DraftablePlayer(Player):
    """
    An inherited class from Player, specialized for pre-season draft analysis.
    This class focuses on season-long projections and risk/reward profiles
    rather than weekly management.
    """

    def __init__(self, fantasy_player, league, extra_players, opponent_schedule, game_lookup, bios, draft_data,
                 game_logs, season, byes):
        # Initialize the base Player class first to set up all common attributes
        super().__init__(fantasy_player, league, extra_players, opponent_schedule, game_lookup, bios, draft_data,
                         game_logs, season, byes)

        # Attributes specific to season-long draft analysis
        self.replacement_ppg = None
        self.season_projection_dist = None

        self.lags = [1, 2, 3]
        self.agg_level = 'season'

    def games_played_mat(self, point_categories):
        """Creates a matrix of historical games for model training."""
        if self.game_logs.empty: return pd.DataFrame([])

        games = self.game_logs[self.game_logs['season'] < self.season]
        if games.empty: return pd.DataFrame([])

        desired_cols = list(point_categories.keys()) + ['receivingTargets', 'gamesPlayed', 'rushingAttempts',
                                                        'passingAttempts', 'applied_total', 'sacks']

        self.rolling_cols = list(set(desired_cols) & set(games.columns))

        grouper = games.groupby('season')

        season_stats = grouper[self.rolling_cols].mean()
        player_info = grouper[games.columns.difference(self.rolling_cols)].first()

        df = pd.concat([player_info, season_stats], axis=1).reset_index(drop=True)
        df['gamesPlayed'] = grouper['gamesPlayed'].sum().values

        df['id'] = self.id
        df['height'] = self.height
        df['weight'] = self.weight
        df['position'] = self.pro_position
        df['college'] = self.college

        if self.draft_info:
            df['draft_pick'] = self.draft_info.overall
            df['draft_grade'] = self.draft_info.pre_draft_grade
            df['pre_draft_ranking'] = self.draft_info.pre_draft_ranking
            df['college_conference'] = self.draft_info.college_conference

        if self.birthday and 'date' in df.columns:
            df['age'] = (df['date'] - self.birthday).dt.days / 365.25
        else:
            df['age'] = None

        lagged_cols = [df]
        for lag in self.lags:
            stat_cols = df[self.rolling_cols].shift(1)
            stat_cols.columns = stat_cols.columns + f'_lag{lag}'

            lagged_cols.append(stat_cols)

        df = pd.concat(lagged_cols, axis=1)

        return df

    def get_future_matchups(self):
        """Generates a list of future games for this player for prediction."""
        rolling_stats = self.get_rolling_stats()

        current_date = datetime(self.season, 9, 1)

        age = (current_date - self.birthday).days / 365.25 if self.birthday else None
        experience = current_date.year - self.first_season if self.first_season else None

        season_info = {
            'name': self.name,
            'season': current_date.year,
            'team': PRO_TEAM_MAP.get(self.pro_team, self.pro_team),
            'player_id': self.id,
            'position': self.pro_position,
            'height': self.height,
            'weight': self.weight,
            'age': age,
            'experience': experience,
            'college': self.college,
            'adp': self.adp.get(self.season),
        }

        if self.draft_info:
            season_info['draft_pick'] = self.draft_info.overall
            season_info['draft_grade'] = self.draft_info.pre_draft_grade
            season_info['pre_draft_ranking'] = self.draft_info.pre_draft_ranking
            season_info['college_conference'] = self.draft_info.college_conference

        season_info.update(rolling_stats)

        return season_info

    def set_projections(self, df, dist_type):
        """Sets future game projections from model output."""

        dist_map = {'gamma': gamma, 'lognormal': lognorm, 'normal': norm}

        dist = dist_map.get(dist_type.lower(), norm)

        self.season_projection_dist = df.set_index(self.agg_level)['params'].map(lambda x: dist(**x)).iloc[0]

        self.projected_ppg = self.season_projection_dist.mean()

        self.ave_projected_ppg = self.projected_ppg.mean()

    def set_vorp(self, replacement_levels):
        """
        Calculates VORP based on the season-long projection distribution.
        This is a risk-neutral calculation based on the mean projection.

        Args:
            replacement_levels (dict): Dictionary mapping position to replacement-level PPG.
        """

        # 1. Determine the correct replacement level PPG for this player
        replacement_ppgs = []
        for position in self.positions:
            if position in ['BE', 'IR'] or position not in replacement_levels:
                continue
            replacement_ppgs.append(replacement_levels.get(position, 0))

        if not replacement_ppgs:
            self.vorp = 0
            return

        self.replacement_ppg = max(replacement_ppgs)

        # 2. Convert to a PPG
        ppg_if_healthy = self.season_projection_dist.mean()

        # 3. Calculate VORP on a per-game basis
        vorp_ppg = ppg_if_healthy - self.replacement_ppg

        # 4. Final VORP is the per-game value multiplied by expected games played
        self.vorp = vorp_ppg * self.ave_play_prob

    def sample_vorp(self, N=1, variates=None):
        """
        Generates N samples of a player's seasonal VORP by simulating both
        performance variance and injury occurrences.
        """
        if self.season_projection_dist is None or self.replacement_ppg is None:
            return np.zeros(N) if N > 1 else 0

        # 1. Sample total points for the season, assuming player is healthy for all games
        if variates is not None:
            ppg_if_healthy_samples = self.season_projection_dist.ppf(variates)
        else:
            ppg_if_healthy_samples = self.season_projection_dist.rvs(size=N, )

        vorps_per_game = ppg_if_healthy_samples - self.replacement_ppg

        # 2. For each sample, simulate how many games are played (not missed)
        if self.ave_play_prob < GAMES_IN_SEASON:
            prob_playing_a_game = self.ave_play_prob or 0
            games_played_samples = np.random.binomial(n=1, p=prob_playing_a_game / GAMES_IN_SEASON, size=(N, GAMES_IN_SEASON))
        else:
            games_played_samples = np.ones((N, GAMES_IN_SEASON))

        return vorps_per_game, games_played_samples