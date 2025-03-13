import os
import numpy as np
import pandas as pd

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

def generate_temperature_data(num_days=30, noise_level=0.5, anomaly_chance=0.1):
    base_temperature = np.linspace(20, 30, num_days)
    noise = np.random.normal(0, noise_level, num_days)
    temperatures = base_temperature + noise

    for i in range(num_days):
        if np.random.rand() < anomaly_chance:
            temperatures[i] += np.random.choice([-10, 10])  # Аномалия: резкое изменение температуры

    return temperatures

for i in range(5):
    temperatures = generate_temperature_data(num_days=30, noise_level=1.0, anomaly_chance=0.2)
    df_train = pd.DataFrame({
        'Day': np.arange(1, 31),
        'Temperature': temperatures
    })
    df_train.to_csv(f'train/temperature_data_{i + 1}.csv', index=False)


for i in range(3):
    temperatures = generate_temperature_data(num_days=30, noise_level=0.5, anomaly_chance=0.1)
    df_test = pd.DataFrame({
        'Day': np.arange(1, 31),
        'Temperature': temperatures
    })
    df_test.to_csv(f'test/temperature_data_{i + 1}.csv', index=False)
print("Данные успешно созданы и сохранены в папках 'train' и 'test'.")
