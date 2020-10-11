import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import bagusformat as bf
from sys import argv

path = 'C:/Users/octavianus.bagus/Documents/Python/kaggle/nwp_data.csv'
#nwp_data = pd.read_csv(path, thousands=r',')
nwp_data = pd.read_csv(path)
nwp_data = nwp_data.dropna(axis=0)
nwp_data = nwp_data.fillna(0)

print(nwp_data.columns)
bf.f01()

#change SPACE to UNDERSCORE in data frame column names; Also Lowercase strings
nwp_data.columns = nwp_data.columns.str.replace(' ', '_')
nwp_data.columns = nwp_data.columns.str.lower()

print(nwp_data.columns)
bf.f01()

y = nwp_data.rental_rate
print(y)
bf.f01()

nama_kolom = []
jumlah_kolom = int(input("Masukkan jumlah kolom : "))
for masukkan in range(jumlah_kolom):
    namadata = str(input("Masukkan nama kolom : "))
    nama_kolom.append(namadata)
print(nama_kolom)
bf.f01()

X = nwp_data[nama_kolom]
print(X)
bf.f01()

print(X.describe())
print(X.head())
bf.f01()

nwp_model = DecisionTreeRegressor(random_state = 1)
print(nwp_model.fit(X, y))
bf.f01()

print(nwp_model.predict(X.head()))
