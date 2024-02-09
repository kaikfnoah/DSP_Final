from pysyncon import Dataprep, Synth, AugSynth, PenalizedSynth
import pandas as pd
from datetime import date, timedelta
from itertools import combinations
import time
import numpy as np
import warnings
from typing import Literal
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_best_synthetic_control_group(treatment_city: str, treatment_date: date, analysis_period: int, prior_period: int,
                                     post_period: int, method: Literal["Standard", "Augmented", "Penalized"],
                                     price_changes_df, reservations_data, min_predictors=3, max_predictors=8, verbose=True):
    syncon_method = method.lower()
    check_syn_con_method(syncon_method)

    # get all possible permutations of all available predictors
    predictors_combinations = get_predictors(min_predictors=min_predictors,
                                             max_predictors=max_predictors)

    # get lists of dates within analysis periods
    full_date_list, prior_date_list, post_date_list = get_date_lists(analysis_period, prior_period, treatment_date)

    # get allowed_cities for each price change
    all_cities = ['Amsterdam', 'Breda', 'Brussels', 'Delft',
                  'Den Bosch', 'Den Haag', 'Enschede', 'Eindhoven',
                  'Groningen', 'Haarlem', 'Nijmegen', 'Rotterdam', 'Tilburg', 'Zwolle']

    control_cities, overlapping_cities, NA_cities = \
        get_control_cities(all_cities, analysis_period, full_date_list, price_changes_df, reservations_data)

    # Raise a ValueError is less than 3 cities are available in control group
    if len(control_cities) < 3:
        error_message = f"Using less than 3 cities (using {len(control_cities)}) is not possible. Please shorten " \
                        f"prior and/or post periods to reduce overlap or avoid missing data. \n" \
                        f"Cities: {overlapping_cities} dropped due to overlapping prices. \n" \
                        f"Cities: {NA_cities} dropped due to missing data in analysis period."
        raise ValueError(error_message)
    # Raise a user warning if fewer than 5 cities are used for synthetic control group creation
    elif len(control_cities) < 5:
        warning_message = f"Only {len(control_cities)} cities used for synthetic control group creation. This warning " \
                          f"does not interfere with the execution."
        warnings.warn(warning_message, UserWarning)

    reservations_data = subset_and_standardize_data(all_cities, full_date_list, reservations_data)

    if verbose:
        print_start_info(NA_cities, overlapping_cities, predictors_combinations, treatment_city, treatment_date)

    best_predictors, total_elapsed_time = best_predictor_search(predictors_combinations, reservations_data,
                                                                treatment_city,
                                                                control_cities, prior_date_list, post_date_list,
                                                                syncon_method, verbose)

    # create dataprep & Synth for best predictors & print outputs
    best_dataprep = get_dataprep(control_cities, best_predictors, prior_date_list, reservations_data, treatment_city)
    synth = create_and_fit_synth_method(syncon_method, best_dataprep)

    # calculate post-treatement fit
    pre_mse = treatment_mse(prior_date_list, reservations_data, synth, treatment_city)
    post_mse = treatment_mse(post_date_list, reservations_data, synth, treatment_city)

    print(f"{treatment_city}:\n"
          f"Best pre-treatment fit found with predictors: {best_predictors}. \n"
          f"Pre-treatment MSE: {round(pre_mse,4)}, Post-treatment MSE: {round(post_mse,4)}. \n"
          f"Total time: {round(total_elapsed_time / 60, 1)}min.")

    # numeric output
    print('\nWeights of used cities:')
    print(synth.weights(round=4))
    print('\nSummary of predictors:')
    print(synth.summary())

    # graphic output
    synth.path_plot(time_period=full_date_list, treatment_time=treatment_date)
    synth.gaps_plot(time_period=full_date_list, treatment_time=treatment_date)

    if isinstance(synth, AugSynth):
        if synth.cv_result is not None:
            synth.cv_result.plot()

    return synth


def get_predictors(min_predictors=1, max_predictors=10):

    if min_predictors < 1:
        raise ValueError("Minimum number of predictors must be at least 1.")
    elif max_predictors > 10:
        raise ValueError("Maximum number of predictors is 10.")

    all_predictors = ['weather_score', 'day_score', 'mist_score', 'rain_score', 'cloud_score', 'wind_score',
                      'feels_like_temp', 'cloud_cover', 'wind_speed', 'sun_hours', 'visibility']
    forecasting_predictors = ['avg_feels_like_temp', 'avg_precipitation', 'rainy_hours', 'min_temp', 'max_temp',
                              'snow_level', 'sun_hours', 'uv_index', 'wind_speed', 'precipitation', 'feels_like_temp',
                              'wind_gust_speed', 'humidity', 'visibility', 'cloud_cover', 'day_length_hours', ]
    # 'vehicle_count_total', 'vehicle_count_active', 'active_customers', 'day_score']
    predictors_combinations = get_combinations(all_predictors, min_predictors, max_predictors)
    return predictors_combinations


def best_predictor_search(predictors_combinations, reservations_data, treatment_city, control_cities, prior_date_list,
                          post_date_list, syncon_method, verbose):
    # iterate through list of predictor_combinations to find best fitting synthetic control group
    i = 0
    best_mse = 9999
    best_predictors = []
    progress_interval = int(np.floor(len(predictors_combinations) / 10))
    predictors_combinations.reverse()  # reverse to start with more complicated fittings (better remaining time estimation)
    start_time = time.time()

    for pred in tqdm(predictors_combinations, disable=not verbose):
        i = i + 1
        #if verbose & (i % progress_interval == 0):
            #print_progress(best_mse, i, predictors_combinations, start_time)

        # new Dataprep object
        dataprep = get_dataprep(control_cities, pred, prior_date_list, reservations_data, treatment_city)

        # synthetic control object
        # Multicollinearity in predictors can lead to singular matrices. These errors are caught here.
        try:
            synth = create_and_fit_synth_method(syncon_method, dataprep)
        except ValueError:
            print(f"Predictors {pred} led to a singular X1 matrix, which cannot be used in the synthetic control method,"
                  f"and are therefore skipped.")
            continue

        # get MSE between treatment & synthetic groups in prior_period
        mse = treatment_mse(prior_date_list, reservations_data, synth, treatment_city)

        # save best mse & predictors
        if mse < best_mse:
            best_mse = mse
            best_predictors = pred

        # output after first iteration to estimate total remaining time
        #if verbose & (i == 1):
            #print_progress(best_mse, i, predictors_combinations, start_time)
    # Output of Iteration:
    total_elapsed_time = round(time.time() - start_time, 0)

    return best_predictors, total_elapsed_time


def subset_and_standardize_data(all_cities, full_date_list, data):
    # subset reservations_data for analysis by relevant cities & dates
    data = data[data['city'].isin(all_cities)]
    data = data[data['date'].isin(full_date_list)]
    # standardize minutes driven by city (so that effects in large and small city may be compared
    # - this is not possible when looking at absolute minutes driven)
    data['standardized_minutes'] = \
        data.groupby('city')['minutes_driven'].transform(lambda x: (x - x.mean()) / x.std())
    return data


def treatment_mse(date_list, reservations_data, synth: Synth, treatment_city: str):
    treatment_minutes = reservations_data[(reservations_data['city'] == treatment_city) &
                                          (reservations_data['date'].isin(date_list))]['standardized_minutes']
    control_minutes = synth._synthetic(time_period=date_list)
    mse = np.mean(np.square(np.array(treatment_minutes) - np.array(control_minutes)))
    return mse


def permutation_testing(synth: Synth,
                        treatment_city: str, treatment_date: date, analysis_period: int, prior_period: int,
                        post_period: int, method: Literal["Standard", "Augmented", "Penalized"],
                        price_changes_df, reservations_data, min_predictors = 3, max_predictors = 8, verbose=True):
    # Do permutation testing by creating a synthetic control group for each control city and visualize the
    # difference in effect

    syncon_method = method.lower()
    check_syn_con_method(syncon_method)

    # get all possible permutations of all available predictors
    predictors_combinations = get_predictors(min_predictors=min_predictors,
                                             max_predictors=max_predictors)

    # get lists of dates within analysis periods
    full_date_list, prior_date_list, post_date_list = get_date_lists(analysis_period, prior_period, treatment_date)

    # get allowed_cities for each price change
    all_cities = ['Amsterdam', 'Breda', 'Brussels', 'Delft',
                  'Den Bosch', 'Den Haag', 'Enschede', 'Eindhoven',
                  'Groningen', 'Haarlem', 'Nijmegen', 'Rotterdam', 'Tilburg', 'Zwolle']

    control_cities, overlapping_cities, NA_cities = \
        get_control_cities(all_cities, analysis_period, full_date_list, price_changes_df, reservations_data)

    reservations_data = subset_and_standardize_data(all_cities, full_date_list, reservations_data)

    # settings for later visualization
    plt.figure(figsize=(12.8, 9.6))
    plt.xticks(rotation=35)

    print(f'\nPermutation Testing for {len(control_cities)} cities:')
    # create synthetic group for each control city & add to visualization
    c=0
    for city in control_cities:
        c = c + 1
        print(f"({c}/{len(control_cities)}) {city}:")
        # create new control_cities list
        new_control_cities = control_cities.copy()
        new_control_cities.append(treatment_city)
        new_control_cities.remove(city)

        best_predictors, total_elapsed_time = best_predictor_search(predictors_combinations, reservations_data,
                                                          treatment_city=city, control_cities=new_control_cities,
                                                          prior_date_list=prior_date_list,
                                                          post_date_list=post_date_list,
                                                          syncon_method=syncon_method,
                                                          verbose=verbose)

        # create dataprep and synth object with best_predictors
        dataprep = get_dataprep(control_cities=new_control_cities, predictors=best_predictors,
                                prior_date_list=prior_date_list, data=reservations_data,
                                treatment_city=city)
        city_synth = create_and_fit_synth_method(syncon_method, dataprep)

        # calculate post-treatement fit
        pre_mse = treatment_mse(prior_date_list, reservations_data, synth, city)
        post_mse = treatment_mse(post_date_list, reservations_data, synth, city)

        print(f"Best pre-treatment fit found with predictors: {best_predictors}. \n"
              f"Pre-treatment MSE: {round(pre_mse,4)}, Post-treatment MSE: {round(post_mse,4)}. \n"
              f"Total time: {round(total_elapsed_time / 60, 1)}min.")

        # only placebo groups for which a good fit (MSE<=1) could be found are added to the plot
        threshold_mse = 1
        if pre_mse <= threshold_mse:
            # new visualization - adapted from pysyncon/base.py
            ts_gap = city_synth._gaps(time_period=full_date_list)
            plt.plot(ts_gap, color="black", linewidth=2, alpha=0.5,
                     label="Cities with Placebo Treatment" if c == 1 else "_n")
            print(f"{city} added as Placebo Group")
        else:
            print(f"Due to a pre-treatment MSE of {round(pre_mse, 4)}, {city} was not added as a Placebo Group. "
                  f"(Pre-treatment MSE <= {threshold_mse} required to be added.)")

    # plotting settings
    # visualization for actual treatment group - adapted from pysyncon/base.py
    tr_gap = synth._gaps(time_period=full_date_list)
    plt.plot(tr_gap, color="red", linewidth=2, label=treatment_city)

    plt.ylabel(synth.dataprep.dependent)
    plt.hlines(
        y=0,
        xmin=min(tr_gap.index),
        xmax=max(tr_gap.index),
        color="black",
        linewidth=2,
        linestyle="dashed",
    )

    plt.axvline(x=treatment_date, ymin=0.05, ymax=0.95, linestyle="dashed")
    plt.title("Difference for Treatment and Placebo Groups", size=26)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.ylabel("Standardized Minutes", size=18)
    plt.grid(True)
    plt.legend(prop={'size': 16})
    plt.show()



def check_syn_con_method(method: str):
    allowed_methods = ['standard', 'augmented', 'penalized']
    if method not in allowed_methods:
        raise ValueError(f"Invalid value for 'syncon_method' attribute. Allowed values are {allowed_methods}.")


def get_dataprep(control_cities, predictors, prior_date_list, data, treatment_city):
    # returns Dataprep object for the given parameters
    return Dataprep(
        foo=data,
        predictors=predictors,
        predictors_op='mean',
        time_predictors_prior=prior_date_list,
        dependent='standardized_minutes',
        unit_variable='city',
        time_variable='date',
        treatment_identifier=treatment_city,
        controls_identifier=control_cities,
        time_optimize_ssr=prior_date_list
    )


def print_start_info(NA_cities, overlapping_cities, predictors_combinations, treatment_city, treatment_date):
    try:
        overlapping_cities.remove(treatment_city)
    except ValueError:
        raise ValueError('No price change found in the selected treatment city. '
                         'Is the treatment date selected correctly? \n'
                         'This may also be a problem caused by missing data.')
    print(f"Determining best set of predictors for {treatment_date} price change in {treatment_city}. "
          f"Total possible combinations of predictors: {len(predictors_combinations)}.")
    print(f"Cities: {overlapping_cities} removed from control group creation due to price changes within the analysis "
          f"period. \n"
          f"Cites: {NA_cities} removed from control group creation due to missing data.")


def get_control_cities(all_cities, analysis_period, full_date_list, price_changes_df, data_df):
    # Returns list of cities to be used for synthetic control group, list of cities with overlapping price changes and
    # list of cities with missing data

    # check cities with overlapping price changes in the analysis period
    overlapping_cities = price_changes_df[(price_changes_df['change_date'].isin(full_date_list)) &
                                          (price_changes_df['city'].isin(all_cities))]['city'].to_list()
    control_cities = all_cities.copy()
    for c in overlapping_cities:
        # this removes all cities with price changes within the analysis period
        # treatment city is also removed since its price change is within the analysis period
        control_cities.remove(c)
    # check for cities with missing data during analysis period
    NA_cities = []
    for city in control_cities:
        if len(data_df[(data_df['city'] == city) & (data_df['date'].isin(full_date_list))]) != analysis_period:
            NA_cities.append(city)
            control_cities.remove(city)
    return control_cities, overlapping_cities, NA_cities


def get_date_lists(analysis_period, prior_period, treatment_date):
    # three lists of date objects, one for the prior, one for the post and one for the full period
    prior_date_list = []
    post_date_list = []
    full_date_list = []
    for x in range(0, analysis_period):
        if x < prior_period:
            prior_date_list.append(treatment_date -
                                   timedelta(days=prior_period) +  # starting_date
                                   timedelta(days=x))
        if x > prior_period:
            post_date_list.append(treatment_date -
                                  timedelta(days=prior_period) +
                                  timedelta(days=x))
        full_date_list.append(treatment_date -
                              timedelta(days=prior_period) +
                              timedelta(days=x))

    return full_date_list, prior_date_list, post_date_list


def get_reservations_data(path_to_local_dir: str, pre_processed_data_file: str):
    # read pre-processed data file
    path = path_to_local_dir + pre_processed_data_file
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d').dt.date
    return data


def get_price_changes(path_to_local_dir: str, price_changes_filename: str):
    # read file for price-change dates
    path = path_to_local_dir + 'price_changes.csv'
    price_changes = pd.read_csv(path)
    price_changes['change_date'] = pd.to_datetime(price_changes['min_date'], format='%Y-%m-%d').dt.date
    return price_changes


def create_and_fit_synth_method(method: str, dataprep: Dataprep):
    # Select method and fit to dataprep object.
    if method == 'standard':
        synth = Synth()
        synth.fit(dataprep=dataprep, optim_method='BFGS', optim_initial='ols', optim_options={'maxiter': 1000})
        return synth
    elif method == 'augmented':
        synth = AugSynth()
        synth.fit(dataprep=dataprep, lambda_=1000)
        return synth
    elif method == 'penalized':
        synth = PenalizedSynth()
        synth.fit(dataprep=dataprep, lambda_=10)
        return synth
    else:
        raise RuntimeError('No applicable method found to create synthetic control group. Please review inputs.')


def print_progress(best_mse, i, predictors_combinations, start_time):
    # Prints progress output on console, including current progress, current best MSE and expected remaining time.
    elapsed_time = time.time() - start_time
    remaining_time = round((len(predictors_combinations) / i * elapsed_time) - elapsed_time, 0)
    print(f'Progress: {round(i / len(predictors_combinations) * 100, 1)}%. Current best MSE: {round(best_mse, 4)}. '
          f'Elapsed time: {round(elapsed_time / 60, 1)}min., Expected remaining time: {round(remaining_time / 60, 1)}min.')


def get_combinations(string_list, min_predictors: int = 1, max_predictors: int = 10):
    all_combinations = []

    # synthetic control methods can make use of a maximum of 10 predictors for fitting,
    # afterwards they create an average across all control groups
    for r in range(min_predictors, max_predictors+1):
        current_combinations = list(combinations(string_list, r))
        all_combinations.extend([list(comb) for comb in current_combinations])

    return all_combinations


if __name__ == "__main__":
    price_change_list = [('Amsterdam', date(2023, 2, 14)),
                         ('Rotterdam', date(2023, 3, 30)),
                         ('Den Haag', date(2023, 3, 20)),
                         ('Groningen', date(2023, 3, 30)),
                         ('Eindhoven', date(2023, 3, 1)),
                         #('Haarlem', date(2023, 3, 1)), #does not work (missing data)
                         ('Tilburg', date(2023, 2, 14)),
                         ('Den Bosch', date(2023, 2, 14)),
                         ('Enschede', date(2023, 3, 30)),
                         ('Brussels', date(2023, 3, 1)),
                         ('Zwolle', date(2023, 3, 20)),
                         ('Nijmegen', date(2023, 3, 1)),
                         #('Breda', date(2023, 2, 15)) #does not work (missing data)
    ]

    for treatment_city, treatment_date in price_change_list:
        method = 'Standard'
        prior_period = 14
        post_period = 14
        analysis_period = prior_period + post_period

        # path to local data (.csv files)
        data_dir = './data/'
        price_change_file = 'price_change_dates.csv'
        reservations_data_file = 'preprocessed_data.csv'

        # get data for price changes
        price_changes_df = get_price_changes(data_dir, price_change_file)

        # get (pre-processed) data for reservations
        reservations_df = get_reservations_data(data_dir, reservations_data_file)

        # find the best synthetic control group
        treatment_synth = get_best_synthetic_control_group(treatment_city, treatment_date,
                                                           analysis_period, prior_period, post_period,
                                                           method,
                                                           price_changes_df, reservations_df,
                                                           min_predictors=3,
                                                           max_predictors=5,
                                                           verbose=True)

        permutation_testing(treatment_synth,
                            treatment_city, treatment_date,
                            analysis_period, prior_period, post_period,
                            method,
                            price_changes_df, reservations_df,
                            min_predictors=3,
                            max_predictors=5,
                            verbose=True)
