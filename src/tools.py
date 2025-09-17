import hashlib
import os
import pickle
import time
from collections import defaultdict
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression

from src.paths import root_dir

# Define a path for your cache files
CACHE_DIR = root_dir / "my_app_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def cache_wrapper(cache_duration=60 * 60 * 24):
    """
    Wrapper function to cache results of the wrapped function.

    Parameters:
    cache_duration (int, optional): Duration in seconds for which the cache is valid.
                                    If None, the cache is considered valid indefinitely.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, force_update=False, **kwargs):
            # Create a unique cache key based on the function name and its arguments
            cache_key = _generate_cache_key(func.__name__, *args, **kwargs)
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

            # Check if we should force an update
            if not force_update and os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    last_updated = cached_data.get("timestamp")
                    if cache_duration is None or (last_updated and time.time() - last_updated < cache_duration):
                        print(f"Using cached data for {func.__name__}.")
                        return cached_data.get("result")
                    else:
                        print(f"Cache expired for {func.__name__}. Recalculating...")

            # Call the original function
            result = func(*args, **kwargs)

            # Save the result to cache with a timestamp
            with open(cache_file, "wb") as f:
                pickle.dump({"timestamp": time.time(), "result": result}, f)

            return result

        return wrapper

    return decorator


def _generate_cache_key(func_name, *args, **kwargs):
    """
    Generates a unique key based on the function name, args, and kwargs.
    """
    key_data = f"{func_name}:{args}:{kwargs}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_best_fas(remaining):
    groups = {}
    for player in sorted(remaining, key=lambda x: x.ave_projected_ppg or 0, reverse=False):
        for pos in player.positions:
            groups[pos] = player.projected_ppg

    return groups


def get_vorp_curve(players, season, visualize=False):  # frac is not needed here
    """
    Generates a VORP vs. ADP curve using Isotonic Regression.
    """
    vorps = []
    adps = []

    for player in players:
        adp = player.adp.get(season)
        vorp = player.vorp
        if None in [adp, vorp]:
            continue
        vorps.append(vorp)
        adps.append(adp)

    if len(adps) < 3:
        print("Not enough data points to generate VORP curve.")
        return None

    # Isotonic Regression doesn't need presorted data, but it's good practice
    # Scikit-learn expects numpy arrays of a specific shape
    adps_np = np.array(adps).reshape(-1, 1)
    vorps_np = np.array(vorps)

    # --- NEW MODEL: ISOTONIC REGRESSION ---
    # We expect a non-increasing (descending) trend.
    # The model finds a non-decreasing (ascending) fit.
    # Trick: Fit the model to the *negative* of the VORP values.
    iso_reg = IsotonicRegression(y_min=None, y_max=None, increasing=False, out_of_bounds="clip")
    iso_reg.fit(adps_np.flatten(), vorps_np)

    # The curve is now the prediction function from the fitted model
    # It handles interpolation automatically.
    curve_func = iso_reg.predict
    # --------------------------------------

    if visualize:
        plt.figure(figsize=(10, 6))
        # Create a smooth line for plotting
        plot_adps = np.linspace(min(adps), max(adps), 500)
        plot_vorps = curve_func(plot_adps)

        plt.scatter(adps, vorps, alpha=0.2, label='Player Data')
        plt.plot(plot_adps, plot_vorps, color='green', linewidth=2, label='Isotonic Regression')

        plt.title(f'VORP vs. ADP Curve ({season}) - Monotonic')
        plt.xlabel('Average Draft Position (ADP)')
        plt.ylabel('Value Over Replacement (VORP)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().invert_xaxis()
        plt.show()

    return curve_func


def get_age_curves(players, frac=0.6, visualize=False):
    """
    Generates model-free aging curves using LOWESS.

    Args:
        players (list): A list of player objects, each with .current_age,
                        .vorp, and .pro_position attributes.
        frac (float, optional): The fraction of data used for smoothing in LOWESS.
                                Larger values create smoother curves. Defaults to 0.6.
        visualize (bool, optional): If True, plots the resulting curves.
                                    Defaults to False.

    Returns:
        dict: A dictionary where keys are positions and values are callable
              functions that return the estimated VORP for a given age.
    """
    ages = defaultdict(list)
    fantasy_points = defaultdict(list)

    # 1. Aggregate data by position (same as before)
    for player in players:

        if 'age' not in player.game_logs.columns or 'applied_total' not in player.game_logs.columns:
            print(f"Skipping player {player.name} due to missing data.")
            continue

        data = player.game_logs[['age', 'applied_total']].dropna()

        if data.empty or data.shape[0] < 1:
            continue

        ages[player.pro_position] += data['age'].tolist()
        fantasy_points[player.pro_position] += data['applied_total'].tolist()

    curves = {}
    for pos in ages:
        # Ensure there's enough data to smooth
        if len(ages[pos]) < 3:
            continue

        # LOWESS requires data to be sorted by the x-values (age)
        pos_data = sorted(zip(ages[pos], fantasy_points[pos]))
        pos_ages, pos_vorps = zip(*pos_data)

        # 2. Apply LOWESS to get smoothed (x, y) points
        # lowess returns an array of [x_smoothed, y_smoothed]
        smoothed_points = sm.nonparametric.lowess(
            pos_vorps, pos_ages, frac=frac
        )

        smoothed_ages = smoothed_points[:, 0]
        smoothed_vorps = smoothed_points[:, 1]

        # 3. Create a callable function using interpolation
        # np.interp will look up an age in smoothed_ages and return the
        # corresponding smoothed_vorps value.
        curves[pos] = lambda x, sa=smoothed_ages, sv=smoothed_vorps: np.interp(x, sa, sv)

    if visualize:
        plt.figure(figsize=(10, 6))
        for pos, curve_func in curves.items():
            # Create a range of ages to plot the curve
            age_space = np.linspace(min(ages[pos]), max(ages[pos]), 100)

            # Get the predicted VORP values from the interpolated function
            pred_vorp = curve_func(age_space)

            # Plot the raw data points as well for comparison
            plt.scatter(ages[pos], fantasy_points[pos], alpha=0.1, label=f'{pos} (data)')
            # Plot the smoothed LOWESS curve
            plt.plot(age_space, pred_vorp, label=f'{pos} (LOWESS)')

        plt.title('Model-Free Aging Curves by Position')
        plt.xlabel('Age')
        plt.ylabel('VORP')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    return curves