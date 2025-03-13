import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib



def prepare_and_train_model(data_folder, model_output_path):
    files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    df_list = []
    for file in files:
        df = pd.read_csv(os.path.join(data_folder, file))
        df_list.append(df)


    full_data = pd.concat(df_list, ignore_index=True)


    X = full_data.drop('Day', axis=1)
    y = full_data['Day']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = LinearRegression()
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Среднеквадратичная ошибка на тестовой выборке: {mse}')

    # Сохранение модели
    joblib.dump(model, model_output_path)
    print(f'Модель сохранена в {model_output_path}')



data_folder = 'train_preprocessed'
model_output_path = 'linear_regression_model.pkl'


prepare_and_train_model(data_folder, model_output_path)

print("Обучение модели завершено.")
