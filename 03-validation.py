import pandas as pd
import bagusformat as bf

melb_path = 'C:/Users/octavianus.bagus/Documents/Python/kaggle/melb_data.csv'
melb_data = pd.read_csv(melb_path)
melb_filtered = melb_data.dropna(axis=0)

y = melb_filtered.Price
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                    'YearBuilt', 'Lattitude', 'Longtitude']
X = melb_filtered[melb_features]

from sklearn.tree import DecisionTreeRegressor
melb_model = DecisionTreeRegressor()
DTR = melb_model.fit(X, y)
print("Describe = ", "\n", X.describe())
print(DTR)
bf.f01()

hd = X.head()
print("Making predictions for the following 5 houses: ")
print(hd)
print("The predictions are (this is still using given data): ")
print(melb_model.predict(hd), "\n")
bf.f01()

print("_________________________________________", "\n")

from sklearn.metrics import mean_absolute_error
predicted_home_prices = melb_model.predict(X)
print("Predicted home prices = ", "\n", predicted_home_prices)

MAE = mean_absolute_error(y, predicted_home_prices)
print("MAE = ", "\n", MAE)
bf.f01()

from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
print(melb_model.fit(train_X, train_y))
bf.f01()

val_predictions = melb_model.predict(val_X)
print("Value predictions : \n", val_predictions)
MAE_2 = mean_absolute_error(val_y, val_predictions)
print("MAE new : \n", MAE_2)
