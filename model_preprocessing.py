import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

    for file in files:

        df = pd.read_csv(os.path.join(input_folder, file))

        scaler = StandardScaler()

        df['Temperature'] = scaler.fit_transform(df[['Temperature']])

        df.to_csv(os.path.join(output_folder, file), index=False)
        print(f'Файл {file} предобработан и сохранен в папке {output_folder}.')


preprocess_data('train', 'train_preprocessed')
preprocess_data('test', 'test_preprocessed')

print("Предобработка данных завершена.")
