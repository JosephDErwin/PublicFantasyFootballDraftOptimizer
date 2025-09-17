import argparse
import multiprocessing

import pandas as pd
from rich.layout import Layout

from src.config import TEAM_ID
from src.paths import data_dir


# --- Worker Functions (Targets for our processes) ---

def monitor_worker(league_id, season_id, team_id, member_id, shared_draft_log, opimizer_trigger, shutdown_event):
    """Worker function to run the DraftMonitor in its own process."""
    print("[Main] Starting Monitor Process...")
    monitor = DraftMonitor(league_id, season_id, team_id, member_id, shared_draft_log, opimizer_trigger, shutdown_event)
    monitor.run()


def optimizer_worker(players, draft_settings, shared_draft_log, results_dict, opimizer_trigger, shutdown_event,
                     vorp_curve, age_curves):
    """Worker function to run the DraftOptimizer in its own process."""
    print("[Main] Starting Optimizer Process...")
    optimizer = DraftOptimizer(players, draft_settings, shared_draft_log, results_dict, opimizer_trigger, shutdown_event,
                               vorp_curve, age_curves)
    optimizer.run()


class TerminalDisplay:
    """
    Manages the live updating terminal display using the 'rich' library.
    """

    def __init__(self):
        self.status = "Initializing..."

    def make_layout(self) -> Layout:
        """Defines the terminal layout."""
        layout = Layout(name="root")
        layout.split_column(
            Layout(Panel("Draft Assistant Status: [bold green]Initializing...[/bold green]", border_style="blue"),
                   name="header", size=3),
            Layout(ratio=1, name="main"),
            Layout(name="footer", size=5)
        )
        layout["main"].split_row(Layout(name="side"), Layout(name="body", ratio=2, minimum_size=60))
        return layout

    def update(self, draft_log, my_team, results_dist, on_the_clock):
        """Updates the tables with the latest data."""
        if on_the_clock:
            self.status = "ON THE CLOCK - Optimizing..."
            status_style = "bold yellow"
        else:
            self.status = "Monitoring draft..."
            status_style = "bold green"

        draft_log_table = Table(title=f"Live Draft Log (Pick {len(draft_log)})")
        draft_log_table.add_column("Pick", justify="right", style="cyan")
        draft_log_table.add_column("Player", style="magenta")
        draft_log_table.add_column("Team", justify="left", style="green")

        for i, pick_info in enumerate(draft_log[-10:]):
            if pick_info['type'] != 'new_pick': continue
            draft_log_table.add_row(str(len(draft_log) - 14 + i), pick_info['player_name'], pick_info['team_name'])

        optimizer_table = Table(title="Optimal Pick Recommendation")
        optimizer_table.add_column("Rank", justify="right", style="cyan")
        optimizer_table.add_column("Player", style="magenta")
        optimizer_table.add_column("Frequency", justify="right", style="green")

        if results_dist is not None and len(results_dist) > 0:
            for i, (player, freq) in enumerate(list(results_dist.items())[:25]):
                optimizer_table.add_row(f"{i + 1}", player, f"{freq:.2%}")
        elif on_the_clock:
            optimizer_table.add_row("1", "Calculating...", "")
        else:
            optimizer_table.add_row("1", "Idle - Waiting for your pick.", "")

        my_team_str = ", ".join(p['player_name'] for p in my_team)

        layout = self.make_layout()
        layout["header"].update(
            Panel(f"Draft Assistant Status: [{status_style}]{self.status}[/{status_style}]", border_style="blue"))
        layout["side"].update(Panel(draft_log_table, title="Recent Picks", border_style="blue"))
        layout["body"].update(Panel(optimizer_table, title="Optimizer", border_style="blue"))
        layout["footer"].update(Panel(my_team_str, title="My Team", border_style="blue"))

        return layout


def main(first_pick, league_id=None, team_id=TEAM_ID):
    """
    The main entry point for the fantasy football draft assistant.
    """
    print("--- Fantasy Football Draft Assistant ---")
    print("Initializing...")

    # TODO: Add in config file loading for draft order and keeper selections

    # --- 1. Pre-load all necessary data ---
    league = get_league(LEAGUE_NAME, SEASON)
    players = get_all_players(league, LEAGUE_NAME)
    my_team_obj = league.get_team_data(team_id)

    print(f"Connected to league '{league.league_id}' for season {SEASON} under team {my_team_obj.team_name}.")

    vorp_curve = get_vorp_curve(players, SEASON)
    age_curves = get_age_curves(players)

    num_rounds = sum(league.roster_slots.values()) - 1

    draft_settings = {
        "my_pick": first_pick,
        "my_team_name": my_team_obj.team_name,
        "rounds": num_rounds,
        "total_picks": num_rounds * len(league.teams),
        "roster_slots": league.roster_slots,
        "teams": len(league.teams),
        "teams_config": [{'owner_name': team.owner} for team in league.teams]
    }
    if league_id is None:
        league_id = league.league_id

    print("Data loading complete.")

    # --- 2. Set up Shared State using a Manager ---
    with multiprocessing.Manager() as manager:
        shared_draft_log = manager.list()  # A process-safe list
        optimization_results = manager.list()  # A process-safe dictionary
        optimization_trigger = manager.Value('i', 0)
        shutdown_event = manager.Event()

        # --- 3. Start the subprocesses ---
        print("Spawning subprocesses...")
        monitor_process = multiprocessing.Process(
            target=monitor_worker,
            args=(league_id, SEASON, team_id, SWID, shared_draft_log, optimization_trigger, shutdown_event),
            name="DraftMonitor"
        )
        optimizer_process = multiprocessing.Process(
            target=optimizer_worker,
            args=(players, draft_settings, shared_draft_log, optimization_results, optimization_trigger, shutdown_event,
                  vorp_curve, age_curves),
            name="DraftOptimizer"
        )

        monitor_process.start()
        time.sleep(15)
        optimizer_process.start()
        print("Monitor and Optimizer processes are running. Press Ctrl+C to exit.")

        # --- 4. Main Application Loop ---
        display = TerminalDisplay()
        try:
            with Live(auto_refresh=False, screen=True, transient=True) as live:
                while not shutdown_event.is_set():
                    # The main loop's only job is to read from shared state and render
                    draft_log_copy = list(shared_draft_log)
                    my_team = [p for p in draft_log_copy if p['type'] == 'new_pick' and p['team_name'] == draft_settings['my_team_name']]
                    on_the_clock = optimization_trigger.value == 1

                    try:
                        results_copy = list(optimization_results)

                        # Convert to a sorted Series for display
                        results_series = pd.Series(results_copy).value_counts(normalize=True).sort_values(ascending=False)
                    except KeyError:
                        continue

                    layout = display.update(draft_log_copy, my_team, results_series, on_the_clock)
                    live.update(layout)
                    live.refresh()

                    time.sleep(2)

        except KeyboardInterrupt:
            print("\nCtrl+C detected. Shutting down gracefully...")
            shutdown_event.set()
        finally:
            # --- 5. Cleanup ---
            print("Waiting for processes to terminate...")
            monitor_process.join(timeout=5)
            optimizer_process.join(timeout=5)

            if monitor_process.is_alive(): monitor_process.terminate()
            if optimizer_process.is_alive(): optimizer_process.terminate()

            print("Application has shut down.")


if __name__ == "__main__":
    TEST_LEAGUE_ID = 69345013
    TEST_TEAM_ID = 8

    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        # This will be raised if the start method has already been set.
        # It's safe to ignore.
        pass
    import multiprocessing
    import time

    import pandas as pd
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    from src.config import LEAGUE_NAME, SEASON, TEAM_ID, SWID
    from src.data.utilities import get_league
    from src.draft.draft_monitor import DraftMonitor
    from src.draft.draft_optimizer import DraftOptimizer
    from src.models.draft_ml import get_all_players
    from src.tools import get_vorp_curve, get_age_curves


    # --- Worker Functions (Targets for our processes) ---

    def monitor_worker(league_id, season_id, team_id, member_id, shared_draft_log, opimizer_trigger, shutdown_event):
        """Worker function to run the DraftMonitor in its own process."""
        print("[Main] Starting Monitor Process...")
        monitor = DraftMonitor(league_id, season_id, team_id, member_id, shared_draft_log, opimizer_trigger,
                               shutdown_event)
        monitor.run()


    def optimizer_worker(players, draft_settings, shared_draft_log, results_dict, opimizer_trigger, shutdown_event,
                         vorp_curve, age_curves):
        """Worker function to run the DraftOptimizer in its own process."""
        print("[Main] Starting Optimizer Process...")
        optimizer = DraftOptimizer(players, draft_settings, shared_draft_log, results_dict, opimizer_trigger,
                                   shutdown_event,
                                   vorp_curve, age_curves)
        optimizer.run()


    class TerminalDisplay:
        """
        Manages the live updating terminal display using the 'rich' library.
        """

        def __init__(self):
            self.status = "Initializing..."

        def make_layout(self) -> Layout:
            """Defines the terminal layout."""
            layout = Layout(name="root")
            layout.split_column(
                Layout(Panel("Draft Assistant Status: [bold green]Initializing...[/bold green]", border_style="blue"),
                       name="header", size=3),
                Layout(ratio=1, name="main"),
                Layout(name="footer", size=5)
            )
            layout["main"].split_row(Layout(name="side"), Layout(name="body", ratio=2, minimum_size=60))
            return layout

        def update(self, draft_log, my_team, results_dist, on_the_clock):
            """Updates the tables with the latest data."""
            if on_the_clock:
                self.status = "ON THE CLOCK - Optimizing..."
                status_style = "bold yellow"
            else:
                self.status = "Monitoring draft..."
                status_style = "bold green"

            draft_log_table = Table(title=f"Live Draft Log (Pick {len(draft_log)})")
            draft_log_table.add_column("Pick", justify="right", style="cyan")
            draft_log_table.add_column("Player", style="magenta")
            draft_log_table.add_column("Team", justify="left", style="green")

            for i, pick_info in enumerate(draft_log[-10:]):
                if pick_info['type'] != 'new_pick': continue
                draft_log_table.add_row(str(len(draft_log) - 14 + i), pick_info['player_name'], pick_info['team_name'])

            optimizer_table = Table(title="Optimal Pick Recommendation")
            optimizer_table.add_column("Rank", justify="right", style="cyan")
            optimizer_table.add_column("Player", style="magenta")
            optimizer_table.add_column("Frequency", justify="right", style="green")

            if results_dist is not None and len(results_dist) > 0:
                for i, (player, freq) in enumerate(list(results_dist.items())[:25]):
                    optimizer_table.add_row(f"{i + 1}", player, f"{freq:.2%}")
            elif on_the_clock:
                optimizer_table.add_row("1", "Calculating...", "")
            else:
                optimizer_table.add_row("1", "Idle - Waiting for your pick.", "")

            my_team_str = ", ".join(p['player_name'] for p in my_team)

            layout = self.make_layout()
            layout["header"].update(
                Panel(f"Draft Assistant Status: [{status_style}]{self.status}[/{status_style}]", border_style="blue"))
            layout["side"].update(Panel(draft_log_table, title="Recent Picks", border_style="blue"))
            layout["body"].update(Panel(optimizer_table, title="Optimizer", border_style="blue"))
            layout["footer"].update(Panel(my_team_str, title="My Team", border_style="blue"))

            return layout


    def main(first_pick, league_id=None, team_id=TEAM_ID, excluded_players=None):
        """
        The main entry point for the fantasy football draft assistant.
        """
        print("--- Fantasy Football Draft Assistant ---")
        print("Initializing...")

        # --- 1. Pre-load all necessary data ---
        league = get_league(LEAGUE_NAME, SEASON)
        players = get_all_players(league, LEAGUE_NAME)
        my_team_obj = league.get_team_data(team_id)

        # Remove excluded players if provided
        if excluded_players is not None:
            init_players = len(players)
            excluded_names = excluded_players['Player Name'].tolist()
            not_found = set(excluded_names) - set([p.name for p in players])

            if not_found:
                print(f"Warning: The following players were not found in the player pool and cannot be excluded: {', '.join(not_found)}")

            players = [p for p in players if p.name not in excluded_names]
            print(f"Excluded {init_players - len(players)} players based on configuration.")

        print(f"Connected to league '{league.league_id}' for season {SEASON} under team {my_team_obj.team_name}.")

        vorp_curve = get_vorp_curve(players, SEASON)
        age_curves = get_age_curves(players)

        num_rounds = sum(league.roster_slots.values()) - 1

        draft_settings = {
            "my_pick": first_pick,
            "my_team_name": my_team_obj.team_name,
            "rounds": num_rounds,
            "total_picks": num_rounds * len(league.teams),
            "roster_slots": league.roster_slots,
            "teams": len(league.teams),
            "teams_config": [{'owner_name': team.owner} for team in league.teams]
        }
        if league_id is None:
            league_id = league.league_id

        print("Data loading complete.")

        # --- 2. Set up Shared State using a Manager ---
        with multiprocessing.Manager() as manager:
            shared_draft_log = manager.list()  # A process-safe list
            optimization_results = manager.list()  # A process-safe dictionary
            optimization_trigger = manager.Value('i', 0)
            shutdown_event = manager.Event()

            # --- 3. Start the subprocesses ---
            print("Spawning subprocesses...")
            monitor_process = multiprocessing.Process(
                target=monitor_worker,
                args=(league_id, SEASON, team_id, SWID, shared_draft_log, optimization_trigger, shutdown_event),
                name="DraftMonitor"
            )
            optimizer_process = multiprocessing.Process(
                target=optimizer_worker,
                args=(players, draft_settings, shared_draft_log, optimization_results, optimization_trigger,
                      shutdown_event,
                      vorp_curve, age_curves),
                name="DraftOptimizer"
            )

            monitor_process.start()
            time.sleep(15)
            optimizer_process.start()
            print("Monitor and Optimizer processes are running. Press Ctrl+C to exit.")

            # --- 4. Main Application Loop ---
            display = TerminalDisplay()
            try:
                with Live(auto_refresh=False, screen=True, transient=True) as live:
                    while not shutdown_event.is_set():
                        # The main loop's only job is to read from shared state and render
                        draft_log_copy = list(shared_draft_log)
                        my_team = [p for p in draft_log_copy if
                                   p['type'] == 'new_pick' and p['team_name'] == draft_settings['my_team_name']]
                        on_the_clock = optimization_trigger.value == 1

                        try:
                            results_copy = list(optimization_results)

                            # Convert to a sorted Series for display
                            results_series = pd.Series(results_copy).value_counts(normalize=True).sort_values(
                                ascending=False)
                        except KeyError:
                            continue

                        layout = display.update(draft_log_copy, my_team, results_series, on_the_clock)
                        live.update(layout)
                        live.refresh()

                        time.sleep(2)

            except KeyboardInterrupt:
                print("\nCtrl+C detected. Shutting down gracefully...")
                shutdown_event.set()
            finally:
                # --- 5. Cleanup ---
                print("Waiting for processes to terminate...")
                monitor_process.join(timeout=5)
                optimizer_process.join(timeout=5)

                if monitor_process.is_alive(): monitor_process.terminate()
                if optimizer_process.is_alive(): optimizer_process.terminate()

                print("Application has shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fantasy Football Draft Assistant")
    parser.add_argument("--league_id", type=int, default=1775728667, help="Your ESPN Fantasy Football league ID.")
    parser.add_argument("--team_id", type=int, default=8, help="Your team ID within the league.")
    parser.add_argument("--first_pick", type=int, default=6,
                        help="The pick number of your first-round pick (e.g., 6 for the 6th pick).")
    args = parser.parse_args()

    excluded_players = pd.read_csv(data_dir / "draft_config.csv", comment="#")

    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        # This will be raised if the start method has already been set.
        # It's safe to ignore.
        pass

    multiprocessing.freeze_support()
    main(
        first_pick=args.first_pick,
        league_id=args.league_id,
        team_id=args.team_id,
        excluded_players=excluded_players
    )