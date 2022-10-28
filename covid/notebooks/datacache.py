import pandas as pd
import numpy as np
import functools
import datetime
import math
from data_processor import DataProcessor

dateformat = "%Y-%m-%d"


# FOLDER_US = "../data/USA/"
# BASE_URL_US = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/" + \
#               "master/csse_covid_19_data/csse_covid_19_time_series/"
# URL_TESTS_US = "https://api.covidtracking.com/v1/states/daily.csv"
# URL_US_States_vaccine = "https://raw.githubusercontent.com/govex/COVID-" + \
#               "19/master/data_tables/vaccine_data/us_data/time_series/vaccine_data_us_timeline.csv"

# globals that need to be set from the notebook
STDT = None
ENDT = None
codes_to_states = None

@functools.cache
def collect_US_data(BASE_URL, FOLDER_US, URL_TESTS_US, URL_US_States_vaccine):
    print("RUNINNG " * 100)
    url_cases_US = BASE_URL + "time_series_covid19_confirmed_US.csv"
    url_deaths_US = BASE_URL + "time_series_covid19_deaths_US.csv"

    df_cases_US = pd.read_csv(url_cases_US, header=0)
    df_deaths_US = pd.read_csv(url_deaths_US, header=0)
    old_dateformat = "%m/%d/%y"
    new_index = [
        datetime.datetime.strptime(d, old_dateformat).strftime(dateformat)
        for d in df_cases_US.columns.values[11:]
    ]
    df_cases_US_States = (
        df_cases_US.groupby(["Province_State"])
        .sum()
        .transpose()
        .drop(["UID", "code3", "FIPS", "Lat", "Long_"])
        .reset_index()
    )
    df_cases_US_States["Date"] = new_index
    df_cases_US_States = (df_cases_US_States
                          .set_index("Date")
                          .drop("index", axis=1))

    df_deaths_US_States = (
        df_deaths_US.groupby(["Province_State"])
        .sum()
        .transpose()
        .drop(["UID", "code3", "FIPS", "Lat", "Long_"])
    )
    pops_US_States = (
        df_deaths_US_States.loc["Population"]
        .to_frame(name="2018")
        .replace({0: math.nan})
        .dropna()
    )

    df_deaths_US_States.drop(["Population"], inplace=True)
    df_deaths_US_States["Date"] = new_index
    df_deaths_US_States = (
        df_deaths_US_States
        .reset_index()
        .set_index("Date")
        .drop("index", axis=1)
    )
    df_rec_tests_US_States = pd.read_csv(
        URL_TESTS_US,
        index_col=[0, 1],
        header=0,
        usecols=["date", "state", "totalTestResults"],
    )

    all_states = df_cases_US_States.columns.values
    df_tests_US_States = (
        pd.pivot_table(df_rec_tests_US_States, index="date", columns="state")
        .loc[:, "totalTestResults"]
        .rename(columns=codes_to_states)
    )
    if df_tests_US_States.shape[0] > len(new_index):
        df_tests_US_States = df_tests_US_States[: len(new_index)]
    df_tests_US_States["Date"] = new_index[: df_tests_US_States.shape[0]]
    df_tests_US_States.set_index("Date", inplace=True)
    df_tests_US_States = df_tests_US_States.loc[STDT:ENDT]
    for c in all_states:
        if c not in df_tests_US_States.columns.values:
            df_tests_US_States[c] = np.nan

    if df_tests_US_States.shape[0] < len(new_index):
        missing_indexes = []
        for d in df_cases_US_States.index.values:
            if d not in df_tests_US_States.index.values:
                missing_indexes.append(d)
        n_rows = len(missing_indexes)
        n_cols = df_tests_US_States.columns.values.shape[0]
        df_tests_US_States = df_tests_US_States.append(
            pd.DataFrame(
                [[np.nan] * n_cols] * n_rows,
                index=missing_indexes,
                columns=df_tests_US_States.columns.values,
            )
        ).sort_index()

    df_vac = pd.read_csv(URL_US_States_vaccine)
    df_vac = pd.pivot_table(
               df_vac.loc[
                       df_vac.loc[:, "Vaccine_Type"] == "All",
                       ["Province_State", "Date", "Doses_admin"],
                   ],
                   values="Doses_admin",
                   index="Date",
                   columns="Province_State",
               )
    df_vac.fillna(method="ffill", inplace=True)
    df_vac = df_vac.loc[STDT:ENDT]

    for c in all_states:
        if c not in df_vac.columns.values:
            df_vac[c] = np.nan

    if df_vac.shape[0] < len(new_index):
        missing_indexes = []
        for d in df_cases_US_States.index.values:
            if d not in df_vac.index.values:
                missing_indexes.append(d)
        n_rows = len(missing_indexes)
        n_cols = df_vac.columns.values.shape[0]
        df_vac = df_vac.append(
            pd.DataFrame(
                [[np.nan] * n_cols] * n_rows,
                index=missing_indexes,
                columns=df_vac.columns.values,
            )
        ).sort_index()

    dict_df_US_States = {
        "Cases": df_cases_US_States,
        "Deaths": df_deaths_US_States,
        "Tests": df_tests_US_States,
        "Vaccine": df_vac,
    }
    for c in list(dict_df_US_States.keys()):
        dict_df_US_States[c] = dict_df_US_States[c].fillna(method="ffill").cummax()

    datasets_US_States = DataProcessor(dict_df_US_States, pops_US_States, "US States")
    # progress_bar.value = 0.7
    # current_action.value = "Collecting US county level Cases/Deaths data..."
    df_cases_US_counties = df_cases_US.drop(
        [
            "UID",
            "iso2",
            "iso3",
            "code3",
            "Country_Region",
            "Lat",
            "Long_",
            "Combined_Key",
        ],
        axis=1,
    )
    df_cases_US_counties.dropna(inplace=True)
    df_cases_US_counties.drop(
        df_cases_US_counties[df_cases_US_counties["Admin2"] == "Unassigned"].index,
        inplace=True,
    )
    df_cases_US_counties.rename(columns={"Admin2": "County"}, inplace=True)

    df_cases_US_counties["County_name"] = (
        df_cases_US_counties["County"] + "_" + df_cases_US_counties["Province_State"]
    )

    df_cases_US_counties.set_index("County_name", inplace=True)
    counties_to_ID = df_cases_US_counties["FIPS"].astype(int).to_dict()
    df_cases_US_counties.drop(
        ["Province_State", "FIPS", "County"], axis=1, inplace=True
    )
    df_cases_US_counties = df_cases_US_counties.transpose()

    df_cases_US_counties["Date"] = new_index
    df_cases_US_counties.set_index("Date", drop=True, inplace=True)

    df_deaths_US_counties = df_deaths_US.drop(
        [
            "UID",
            "iso2",
            "iso3",
            "code3",
            "Country_Region",
            "Lat",
            "Long_",
            "Combined_Key",
        ],
        axis=1,
    )
    df_deaths_US_counties.dropna(inplace=True)
    df_deaths_US_counties.drop(
        df_deaths_US_counties[df_deaths_US_counties["Admin2"] == "Unassigned"].index,
        inplace=True,
    )
    df_deaths_US_counties.rename(columns={"Admin2": "County"}, inplace=True)

    df_deaths_US_counties["County_name"] = (
        df_deaths_US_counties["County"] + "_" + df_deaths_US_counties["Province_State"]
    )

    df_deaths_US_counties.set_index("County_name", inplace=True)
    pops_US_counties = (
        df_deaths_US_counties.loc[:, "Population"]
        .to_frame(name="2018")
        .replace({0: math.nan})
        .dropna()
        .replace({0: math.nan})
    )
    df_deaths_US_counties.drop(
        ["Province_State", "FIPS", "County", "Population"], axis=1,
        inplace=True
    )
    df_deaths_US_counties = df_deaths_US_counties.transpose()

    df_deaths_US_counties["Date"] = new_index
    df_deaths_US_counties.set_index("Date", drop=True, inplace=True)
    
    nan_data = np.empty(df_cases_US_counties.shape)
    nan_data[:] = np.nan
    df_tests_US_counties = pd.DataFrame(
        nan_data,
        index=df_cases_US_counties.index.values,
        columns=df_cases_US_counties.columns.values,
    )
    
    df_vac_US_counties = pd.DataFrame(
        nan_data,
        index=df_cases_US_counties.index.values,
        columns=df_cases_US_counties.columns.values,
    )
    
    dict_df_US_counties = {
        "Cases": df_cases_US_counties,
        "Deaths": df_deaths_US_counties,
        "Tests": df_tests_US_counties,
        "Vaccine": df_vac_US_counties,
    }
    for c in list(dict_df_US_counties.keys()):
        dict_df_US_counties[c] = dict_df_US_counties[c].fillna(method="ffill").cummax()
        
    datasets_US_counties = DataProcessor(
        dict_df_US_counties, pops_US_counties, "US Counties", counties_to_ID
    )

    return datasets_US_States, datasets_US_counties


@functools.cache
def collect_World_data(BASE_URL, FOLDER_WORLD, URL_TESTS, URL_vaccine):
    pops = pd.read_csv(FOLDER_WORLD + "Population.csv", header=0).dropna()
    pops.set_index("Country Code", inplace=True) 
    url_cases = BASE_URL + "time_series_covid19_confirmed_global.csv"
    url_deaths = BASE_URL + "time_series_covid19_deaths_global.csv"
    dataframes = {
        "Cases": pd.read_csv(url_cases, header=0),
        "Deaths": pd.read_csv(url_deaths, header=0),
    }
    for key, df in dataframes.items():
        df.loc[df["Province/State"] == "Hong Kong", "Country/Region"] = \
            "Hong Kong"
        df.set_index("Country/Region", inplace=True)
    country_codes = pd.read_csv(
        FOLDER_WORLD + "world_map_codes.csv", header=0, index_col=0
    )
    countries_to_codes = country_codes["ISOA3"].to_dict()
    new_names = {
        "Antigua and Barbuda": "Antigua & Barbuda",
        "Bosnia and Herzegovina": "Bosnia",
        "Cabo Verde": "Cape Verde",
        "Congo (Brazzaville)": "Congo - Brazzaville",
        "Congo (Kinshasa)": "Congo - Kinshasa",
        "Eswatini": "Swaziland",
        "Holy See": "Vatican City",
        "Korea, South": "South Korea",
        "North Macedonia": "Macedonia",
        "Saint Lucia": "St. Lucia",
        "Saint Vincent and the Grenadines": "St. Vincent & Grenadines",
        "Trinidad and Tobago": "Trinidad & Tobago",
        "US": "United States",
        "Saint Kitts and Nevis": "St. Kitts & Nevis",
        "Burma": "Myanmar",
        "Taiwan*": "Taiwan",
    }
    # normalize country names
    for df in dataframes.values():
        df.rename(index=new_names, inplace=True)

    # assign dummy codes to a few entities
    countries_to_codes["World"] = "WLD"
    countries_to_codes["Diamond Princess"] = "DPS"
    countries_to_codes["Taiwan"] = "TWN"
    countries_to_codes["West Bank and Gaza"] = "PSE"
    countries_to_codes["Kosovo"] = "XKX"
    countries_to_codes["MS Zaandam"] = "MSZ"
    codes_to_countries = {v: k for k, v in countries_to_codes.items()}
    for df in dataframes.values():
        df["code"] = [
            countries_to_codes[cod] if cod in countries_to_codes.keys()
            else None for cod in df.index.values
        ]
    
    old_dateformat = "%m/%d/%y"
    new_index = [
        datetime.datetime.strptime(d, old_dateformat).strftime(dateformat)
        for d in dataframes["Cases"].columns.values[3:-1]
    ]
    # set index and dateformat
    for k, df in dataframes.items():
        df = df.iloc[:, 3:].groupby(["code"]).sum().transpose().reset_index()
        df["Date"] = new_index
        dataframes[k] = df.set_index("Date", drop=True).drop("index", axis=1)

    df_world_tests = pd.read_csv(
        URL_TESTS,
        index_col=0,
        header=0,
        usecols=["ISO code", "Date", "Cumulative total"],
    )
    df_world_tests = (
        pd.pivot_table(df_world_tests, index="Date", columns="ISO code")
        .loc[:, "Cumulative total"]
        .fillna(method="ffill")
    )
    for c in dataframes["Cases"].columns.values:
        if c not in df_world_tests.columns.values:
            df_world_tests[c] = np.nan
    STDT = dataframes["Cases"].index[0]
    ENDT = dataframes["Cases"].index[-1]
    df_world_tests = df_world_tests.loc[STDT:ENDT]

    if df_world_tests.shape[0] < len(new_index):
        missing_indexes = []
        for d in dataframes["Cases"].index.values:
            if d not in df_world_tests.index.values:
                missing_indexes.append(d)
        n_rows = len(missing_indexes)
        n_cols = df_world_tests.columns.values.shape[0]
        df_world_tests = df_world_tests.append(
            pd.DataFrame(
                [[np.nan] * n_cols] * n_rows,
                index=missing_indexes,
                columns=df_world_tests.columns.values,
            )
        ).sort_index()

    dataframes["Tests"] = df_world_tests.loc[:, dataframes["Cases"].columns.values]

    df_vac = pd.read_csv(
        URL_vaccine,
        usecols=["date", "iso_code", "total_vaccinations"],
        index_col=[0, 1],
    )
    df_vac = pd.pivot_table(df_vac, columns=["iso_code"], index=["date"]).loc[
        STDT:ENDT, "total_vaccinations"
    ]
    df_vac.fillna(method="ffill", inplace=True)

    for c in dataframes["Cases"].columns.values:
        if c not in df_vac.columns.values:
            df_vac[c] = np.nan

    if df_vac.shape[0] < len(new_index):
        missing_indexes = []
        for d in dataframes["Cases"].index.values:
            if d not in df_vac.index.values:
                missing_indexes.append(d)
        n_rows = len(missing_indexes)
        n_cols = df_vac.columns.values.shape[0]
        df_vac = df_vac.append(
            pd.DataFrame(
                [[np.nan] * n_cols] * n_rows,
                index=missing_indexes,
                columns=df_vac.columns.values,
            )
        ).sort_index()

    dataframes["Vaccine"] = df_vac.loc[STDT:ENDT, dataframes["Cases"].columns.values]
    
    for c in list(dataframes.keys()):
        dataframes[c] = dataframes[c].fillna(method="ffill").cummax()
        
    datasets_World = DataProcessor(
        dataframes, pops, "World", add_world_data=True
    )  # World data processor
    # progress_bar.value = 0.5
    # current_action.value = (
    #     "Collecting US state level Cases/Deaths/Vaccine/Test data..."
    # )
    return datasets_World, STDT, ENDT, countries_to_codes, codes_to_countries