from sklearn.preprocessing import MinMaxScaler

# Your known minimal and maximal values
min_value = 10
max_value = 50

# Instantiate MinMaxScaler with specified feature_range
custom_scaler = MinMaxScaler(feature_range=(min_value, max_value))

# Now, you can use this custom_scaler to transform new data without fitting
new_data = [[15], [25], [35]]
scaled_data = custom_scaler.transform(new_data)

print(scaled_data)
