import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib


# Функция для тестирования модели
def test_model(test_data_folder, model_path):
    # Получение списка файлов в папке
    files = [f for f in os.listdir(test_data_folder) if f.endswith('.csv')]

    # Объединение всех файлов в один DataFrame
    df_list = []
    for file in files:
        df = pd.read_csv(os.path.join(test_data_folder, file))
        df_list.append(df)

    # Объединение всех данных в один DataFrame
    test_data = pd.concat(df_list, ignore_index=True)

    # Разделение данных на признаки (X) и целевую переменную (y)
    X_test = test_data.drop('Day', axis=1)  # Предполагается, что целевая переменная называется 'Target'
    y_test = test_data['Day']

    # Загрузка модели
    model = joblib.load(model_path)

    # Прогнозирование
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    print(f'Среднеквадратичная ошибка на тестовой выборке: {mse}')


# Путь к папке с предобработанными тестовыми данными и путь к сохраненной модели
test_data_folder = 'test_preprocessed'
model_path = 'linear_regression_model.pkl'

# Тестирование модели
test_model(test_data_folder, model_path)

print("Тестирование модели завершено.")
