import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the dataset
df = pd.read_csv("data.csv")

# Convert lap time to seconds
df['official_lap_time'] = pd.to_timedelta('00:' + df['official_lap_time']).dt.total_seconds()

# Extract features and target variable
X = df.drop(['car_name', 'official_lap_time'], axis=1)
y = df['official_lap_time']

# Define the column transformer for preprocessing
categorical_features = ['drive_type', 'cylinder_layout', 'induction_type', 'gearbox_type', 'tires_used']
numeric_features = list(set(X.columns) - set(categorical_features))

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the regressor pipeline
regressor_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())  # You can replace this with any regression model
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters to search
param_grid = {
    'regressor__n_estimators': [10, 20, 30, 40, 50, 100, 150, 200],
    'regressor__max_depth': [None, 10, 20, 30],
}

# Perform GridSearchCV to find the best model and hyperparameters
grid_search = GridSearchCV(regressor_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Load new car features from CSV
new_car_features_df = pd.read_csv("input_data.csv")

# Predict lap times for new cars
new_car_features_df['predicted_lap_time_seconds'] = best_model.predict(new_car_features_df)

# Use KNN to find cars with similar lap times
knn_features = X[['engine_output_ps', 'car_weight', 'top_speed', '0_100_kmh_time', 'tyre_size_inches', 'ground_clearance']]
knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_model.fit(knn_features.values)  # Provide feature values without feature names

# Find cars with similar lap times for each new car
similar_cars_indices = knn_model.kneighbors(new_car_features_df[['engine_output_ps', 'car_weight', 'top_speed', '0_100_kmh_time', 'tyre_size_inches', 'ground_clearance']], return_distance=False)
similar_cars_list = []
for indices in similar_cars_indices:
    similar_cars = df.iloc[indices][['car_name', 'official_lap_time']]
    similar_cars_list.append(similar_cars.reset_index(drop=True))

# Use k-means clustering to identify the class of the new car
kmeans_features = X[['engine_output_ps', 'car_weight', 'top_speed', '0_100_kmh_time', 'tyre_size_inches', 'ground_clearance']]
kmeans_model = KMeans(n_clusters=10, random_state=42, n_init='auto')  # Adjust the number of clusters as needed
df['cluster'] = kmeans_model.fit_predict(kmeans_features)

# Predict the cluster for each new car
new_car_features_df['cluster'] = kmeans_model.predict(new_car_features_df[['engine_output_ps', 'car_weight', 'top_speed', '0_100_kmh_time', 'tyre_size_inches', 'ground_clearance']])

# Get the average lap time of cars in the same cluster for each new car
average_lap_time_cluster_list = []
for cluster in new_car_features_df['cluster']:
    average_lap_time_cluster = df[df['cluster'] == cluster]['official_lap_time'].mean()
    average_lap_time_cluster_list.append(average_lap_time_cluster)

# Create a DataFrame to store results
results_df = new_car_features_df.copy()
results_df['similar_cars'] = similar_cars_list
results_df['average_lap_time_in_cluster'] = average_lap_time_cluster_list

# Save results to a CSV file
results_df.to_csv("predicted_lap_times.csv", index=False)