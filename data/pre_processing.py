import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

user = "private information" # unable to share this information
password = "private information" # unable to share this information

conn_str = "private information" # unable to share this information

OLD_PRICE = 0.248
NEW_PRICE = 0.273

CITY_TO_COUNTRY = {
    'Amsterdam': 'NL',
    'Rotterdam': 'NL',
    'Den Haag': 'NL',
    'Brussels': 'BE',
    'Groningen': 'NL',
    'Delft': 'NL',
    'Eindhoven': 'NL',
    'Haarlem': 'NL',
    'Den Bosch': 'NL',
    'Düsseldorf': 'DE',
    'Hamburg': 'DE', 
    'Berlin': 'DE', 
    'Enschede': 'NL', 
    'Aachen': 'DE', 
    'Zwolle': 'NL', 
    'Tilburg': 'NL',
    'Nijmegen': 'NL',
    'Breda': 'NL',
    'Elite': np.nan,
}

QUERY = "private information" # unable to share this information

def query_reservations(query):
    return pd.read_sql(query, conn_str)

def fix_price_change_data(df):
    cities = list(set(df['city']))
    for city in cities:
        
        # Haarlem has a price change error in a row
        if city == 'Haarlem':
            wrong_date = df[(df['city'] == 'Haarlem') &
                            (df['price'] == NEW_PRICE)
                            ]['date'].min()
            row_to_remove = df[(df['city'] == 'Haarlem') & 
                            (df['date'] == wrong_date) & 
                            (df['price'] == NEW_PRICE)
                            ].index.values.tolist()
            df.drop(row_to_remove, inplace=True)
        
        # Rows that are not either one of the standard prices
        rows_to_remove = df[(df['price'] != OLD_PRICE) & 
                            (df['price'] != NEW_PRICE)
                            ].index.values.tolist()
        df.drop(rows_to_remove, inplace=True)
        
        # Rows with the old price that overlap with the new price
        price_change_date = df[(df['city'] == city) &
                               (df['price'] == NEW_PRICE)
                               ]['date'].min()
        rows_to_remove = df[(df['city'] == city) & 
                            (df['date'] > price_change_date) & 
                            (df['price'] == OLD_PRICE)
                            ].index.values.tolist()
        
        # Rows to merge
        rows_to_merge = df[(df['city'] == city) & 
                            (df['date'] == price_change_date)
                            ].index.values.tolist()
        
        if len(rows_to_merge) > 1:
            df.loc[rows_to_merge[0], 'minutes_driven'] += df.loc[rows_to_merge[1], 'minutes_driven']
            df.drop(rows_to_merge[1], inplace=True)

        df.drop(rows_to_remove, inplace=True)
    
    return df


if __name__ == "__main__":

    # Raw data from the reservations table
    df = pd.DataFrame(query_reservations(QUERY))

    # Create a new 'country' column based on the 'city' column
    df['country'] = df['city'].map(CITY_TO_COUNTRY)

    # Weather score nulls are dropped (mainly in 2017); price nulls are dropped (forecasted entries of 2024)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Through research, we found that Hamburg has a lot of null values for the weather data, making it impossible
    # to work with.
    rows_to_drop = df['city']  == 'Hamburg'
    df = df[~rows_to_drop]

    # Drop rows with overlapping prices (in the same city)
    df = fix_price_change_data(df)
    
    # Add date information 
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.day_name()
    df['week_number'] = df['date'].dt.isocalendar()['week']

    # Add daily scores (expressed in significance per week)
    df['week_total'] = 0
    for city in set(df['city']):
        for week in set(df[df['city'] == city]['week_number']):
            for year in set(df[(df['city'] == city) & (df['week_number'] == week)]['year']):
                week_total = sum(df[(df['week_number'] == week) & (df['city'] == city) & (df['year'] == year)]['minutes_driven'])
                conditions = (df['week_number'] == week) & (df['city'] == city) & (df['year'] == year)
                df.loc[conditions, 'week_total'] = week_total
    df['day_score'] = df['minutes_driven'] / df['week_total']

    # Reorder columns
    column_order = ['date', 'city', 'week_number', 'day_of_week', 'day_score', 'country', 'price', 'minutes_driven', 'weather_score', 
                    'mist_score', 'rain_score', 'cloud_score', 'wind_score', 'feels_like_temp', 'cloud_cover', 'wind_speed', 
                    'sun_hours', 'visibility']
    df = df[column_order]

    # Add index
    df.set_index(['city', 'date'], inplace=True)

    # Write df to file
    df.to_csv('preprocessed_data.csv')