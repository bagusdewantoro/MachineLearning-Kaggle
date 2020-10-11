import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import bagusformat as bf

path = 'C:/Users/octavianus.bagus/Documents/Python/kaggle/melb_data.csv'
melb_data = pd.read_csv(path)

show = melb_data.columns
print(show)
bf.f01()

y = melb_data.Price
print(y)
bf.f01()

features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_data[features]
print(X)
bf.f01()

dscr = X.describe()
print(dscr)
bf.f01()

hd = X.head()
print(hd)
bf.f01()

print("\
============================  model using scikit-learn ==========================\n")

melb_model = DecisionTreeRegressor(random_state = 1)

DTR = melb_model.fit(X, y)
print(DTR)
bf.f01()

print("Making predictions for the following 5 houses: ")
print(hd)
print("The predictions are (this is still using given data): ")
print(melb_model.predict(hd), "\n")
