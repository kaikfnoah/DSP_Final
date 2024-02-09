import pandas as pd

df = pd.read_csv('our_preds.csv')
df = df.groupby(['city', 'price']).date.agg(['min', 'max'])

df.to_csv('price_change_dates.csv')