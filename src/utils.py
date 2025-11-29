import pandas as pd
import numpy as np

def load_dataset(path):
    """Загрузка датасета"""
    return pd.read_csv(path)

def print_dataset_info(df):
    """Вывод информации о датасете"""
    print(f"Размер датасета: {df.shape}")
    print(f"Пропущенные значения: {df.isnull().sum().sum()}")
    print(f"Столбцы: {df.columns.tolist()}")
    
def save_results_to_csv(results, filename):
    """Сохранение результатов в CSV"""
    pd.DataFrame(results).to_csv(filename, index=False)
