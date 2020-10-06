import pandas as pd
import bagusformat as bf

melb_path = 'C:/Users/octavianus.bagus/Documents/Python/kaggle/melb_data.csv'

melb_data = pd.read_csv(melb_path)
print(melb_data)

bf.f01()

show = melb_data.describe()
print(show)
