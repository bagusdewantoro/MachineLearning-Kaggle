import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import bagusformat as bf
from sys import argv

path = 'C:/Users/octavianus.bagus/Documents/Python/kaggle/nwp_inv.csv'
nwp_data = pd.read_csv(path)

print(nwp_data.columns)
bf.f01()

#change SPACE to UNDERSCORE in data frame column names
nwp_data.columns = nwp_data.columns.str.replace(' ', '_')
nwp_data.columns = nwp_data.columns.str.lower()

print(nwp_data.columns)
bf.f01()

y = nwp_data.Rental_Rate
print(y)
bf.f01()

nama_kolom = []
jumlah_kolom = int(input("Masukkan jumlah kolom : "))
for masukkan in range(jumlah_kolom):
    namadata = str(input())
    nama_kolom.append(namadata)
#angka = 0
#jumlah_kolom = int(input("Masukkan jumlah kolom : "))
#for masukkan in range(jumlah_kolom):
#    data1 = input("Input data1 : ")

X = nwp_data[nama_kolom]
print(X)
bf.f01()
