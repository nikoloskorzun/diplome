import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import timeseries_dataset_from_array

class MultivariateTimeSeriesPreprocessor:
    """
    Предобработчик многомерных временных рядов для GAN-обнаружения аномалий
    """
    
    def __init__(self, sequence_length=50, stride=1, scaling_method='minmax'):
        """
        Инициализация предобработчика
        
        Параметры:
        - sequence_length: длина временного окна
        - stride: шаг скользящего окна
        - scaling_method: метод нормализации ('minmax', 'standard', 'robust')
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.scaling_method = scaling_method
        self.scalers = {}
        self.feature_names = None
        self.is_fitted = False
        
    def fit_scalers(self, data):
        """
        Обучение скалеров на данных
        
        Параметры:
        - data: DataFrame или numpy array с временными рядами
        """
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns.tolist()
            data_array = data.values
        else:
            data_array = data
            self.feature_names = [f'feature_{i}' for i in range(data_array.shape[1])]
        
        # Обучение отдельного скалера для каждой фичи
        for i, feature_name in enumerate(self.feature_names):
            if self.scaling_method == 'minmax':
                scaler = MinMaxScaler(feature_range=(-1, 1))
            elif self.scaling_method == 'standard':
                scaler = StandardScaler()
            else:
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            scaler.fit(data_array[:, i].reshape(-1, 1))
            self.scalers[feature_name] = scaler
        
        self.is_fitted = True
        return self
    
    def transform(self, data):
        """
        Применение преобразований к данным
        
        Параметры:
        - data: DataFrame или numpy array
        
        Возвращает:
        - нормализованные данные
        """
        if not self.is_fitted:
            raise ValueError("Сначала необходимо обучить скалеры методом fit_scalers()")
        
        if isinstance(data, pd.DataFrame):
            data_array = data[self.feature_names].values
        else:
            data_array = data
        
        # Нормализация каждой фичи
        scaled_data = np.zeros_like(data_array)
        for i, feature_name in enumerate(self.feature_names):
            scaled_data[:, i] = self.scalers[feature_name].transform(
                data_array[:, i].reshape(-1, 1)
            ).flatten()
        
        return scaled_data
    
    def fit_transform(self, data):
        """
        Обучение скалеров и применение преобразований
        """
        self.fit_scalers(data)
        return self.transform(data)
    
    def inverse_transform(self, scaled_data):
        """
        Обратное преобразование нормализованных данных
        
        Параметры:
        - scaled_data: нормализованные данные
        
        Возвращает:
        - исходные данные
        """
        if not self.is_fitted:
            raise ValueError("Сначала необходимо обучить скалеры")
        
        original_data = np.zeros_like(scaled_data)
        for i, feature_name in enumerate(self.feature_names):
            original_data[:, i] = self.scalers[feature_name].inverse_transform(
                scaled_data[:, i].reshape(-1, 1)
            ).flatten()
        
        return original_data
    
    def create_sequences(self, data, target_data=None):
        """
        Создание последовательностей для временных рядов
        
        Параметры:
        - data: входные данные
        - target_data: целевые данные (если отличаются от входных)
        
        Возвращает:
        - X: входные последовательности (batch_size, sequence_length, n_features)
        - y: целевые последовательности (если указаны)
        """
        if isinstance(data, pd.DataFrame):
            data_array = data[self.feature_names].values
        else:
            data_array = data
        
        if target_data is not None:
            if isinstance(target_data, pd.DataFrame):
                target_array = target_data[self.feature_names].values
            else:
                target_array = target_data
        else:
            target_array = data_array
        
        # Создание последовательностей
        X, y = [], []
        
        for i in range(0, len(data_array) - self.sequence_length + 1, self.stride):
            X.append(data_array[i:(i + self.sequence_length)])
            y.append(target_array[i:(i + self.sequence_length)])
        
        return np.array(X), np.array(y)
    
    def prepare_data_for_gan(self, data, test_size=0.2, validation_size=0.1):
        """
        Полная подготовка данных для GAN
        
        Параметры:
        - data: исходные данные
        - test_size: доля тестовых данных
        - validation_size: доля валидационных данных
        
        Возвращает:
        - train_data, val_data, test_data: подготовленные данные
        - train_sequences, val_sequences, test_sequences: последовательности
        """
        # Нормализация данных
        scaled_data = self.fit_transform(data)
        
        # Разделение на train/test
        n_samples = len(scaled_data)
        n_test = int(n_samples * test_size)
        n_val = int((n_samples - n_test) * validation_size)
        
        if n_test == 0:
            test_data = scaled_data[-100:] if len(scaled_data) > 100 else scaled_data
            train_val_data = scaled_data[:-100] if len(scaled_data) > 100 else scaled_data
        else:
            test_data = scaled_data[-n_test:]
            train_val_data = scaled_data[:-n_test]
        
        if n_val == 0:
            val_data = train_val_data[-50:] if len(train_val_data) > 50 else train_val_data
            train_data = train_val_data[:-50] if len(train_val_data) > 50 else train_val_data
        else:
            val_data = train_val_data[-n_val:]
            train_data = train_val_data[:-n_val]
        
        # Создание последовательностей
        train_sequences, _ = self.create_sequences(train_data)
        val_sequences, _ = self.create_sequences(val_data)
        test_sequences, _ = self.create_sequences(test_data)
        
        print(f"Размеры данных:")
        print(f"  Train: {train_data.shape} -> {train_sequences.shape}")
        print(f"  Validation: {val_data.shape} -> {val_sequences.shape}")
        print(f"  Test: {test_data.shape} -> {test_sequences.shape}")
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_sequences': train_sequences,
            'val_sequences': val_sequences,
            'test_sequences': test_sequences,
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }
