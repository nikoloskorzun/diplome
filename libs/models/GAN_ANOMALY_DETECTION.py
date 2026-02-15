import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, regularizers
from sklearn.model_selection import train_test_split
import itertools
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
def preprocess_data(data: np.ndarray, sequence_length: int, stride: int = 1) -> np.ndarray:
    """
    Преобразует многомерный временной ряд в последовательности окон.
    
    Args:
        data: Многомерный временной ряд (n_samples, n_features)
        sequence_length: Длина окна
        stride: Шаг сдвига окна (по умолчанию 1)
    
    Returns:
        np.ndarray: Данные в формате (n_windows, sequence_length, n_features)
    """
    n_samples, n_features = data.shape
    n_windows = (n_samples - sequence_length) // stride + 1
    X = np.zeros((n_windows, sequence_length, n_features))
    
    for i in range(n_windows):
        start = i * stride
        end = start + sequence_length
        X[i] = data[start:end]
    
    return X
def create_generator(
    latent_dim: int,
    sequence_length: int,
    n_features: int,
    gen_layers: int,
    gen_layer_types: List[str],  
    gen_layer_sizes: List[int],
    activation: str,
    l2_reg: float,
    dropout: float,
    use_batch_norm: bool = False,
    kernel_initializer: str = 'glorot_uniform',
    use_skip_connections: bool = False,
    **argv
) -> keras.Model:
    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = inputs
    skip_connections = []
    
    # Начальная обработка для первого слоя
    current_dim = 2  # Начинаем с 2D (latent_dim)
    
    for i in range(gen_layers):
        # Определяем текущую размерность и преобразуем при необходимости
        if gen_layer_types[i] == 'Dense':
            # Если вход 3D, преобразуем в 2D
            if current_dim == 3:
                x = tf.keras.layers.Flatten()(x)
                current_dim = 2
            
            x = tf.keras.layers.Dense(
                gen_layer_sizes[i],
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            current_dim = 2
            
        elif gen_layer_types[i] == 'LSTM':
            # Если вход 2D, преобразуем в 3D
            if current_dim == 2:
                # Добавляем временное измерение
                x = tf.keras.layers.Reshape((1, gen_layer_sizes[i-1] if i > 0 else latent_dim))(x)
                current_dim = 3
            
            x = tf.keras.layers.LSTM(
                gen_layer_sizes[i],
                activation=activation,
                return_sequences=(i < gen_layers - 1),
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            current_dim = 3 if (i < gen_layers - 1) else 2
            
        elif gen_layer_types[i] == 'GRU':
            if current_dim == 2:
                x = tf.keras.layers.Reshape((1, gen_layer_sizes[i-1] if i > 0 else latent_dim))(x)
                current_dim = 3
            
            x = tf.keras.layers.GRU(
                gen_layer_sizes[i],
                activation=activation,
                return_sequences=(i < gen_layers - 1),
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            current_dim = 3 if (i < gen_layers - 1) else 2
            
        elif gen_layer_types[i] == 'Conv1DTranspose':
            # Conv1DTranspose всегда требует 3D
            if current_dim != 3:
                # Создаем временное измерение
                prev_size = gen_layer_sizes[i-1] if i > 0 else latent_dim
                x = tf.keras.layers.Reshape((1, prev_size))(x)
                current_dim = 3
            
            # Если последний слой - Conv1DTranspose, то настраиваем размерность
            if i == gen_layers - 1:
                x = tf.keras.layers.Conv1DTranspose(
                    filters=n_features,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    activation='tanh'
                )(x)
            else:
                x = tf.keras.layers.Conv1DTranspose(
                    filters=gen_layer_sizes[i],
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    activation=activation,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    kernel_initializer=kernel_initializer
                )(x)
            current_dim = 3
            
        # Добавляем BatchNorm и Dropout
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
        
        # Skip connections
        if use_skip_connections and i > 0:
            # Проверяем, что в списке достаточно элементов
            if len(skip_connections) >= 1:
                # Соединяем с последним элементом в списке (предыдущим слоем)
                if x.shape[1:] == skip_connections[-1].shape[1:]:
                    x = tf.keras.layers.Add()([x, skip_connections[-1]])
        
        skip_connections.append(x)
    
    # Убедимся, что последний слой имеет правильную размерность
    if current_dim == 2:
        # Если остался 2D, преобразуем в последовательность
        x = tf.keras.layers.Dense(
            sequence_length * n_features,
            activation='tanh',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = tf.keras.layers.Reshape((sequence_length, n_features))(x)
    
    return models.Model(inputs, x)

def create_discriminator(
    sequence_length: int,
    n_features: int,
    disc_layers: int,
    disc_layer_types: List[str],
    disc_layer_sizes: List[int],
    activation: str,
    l2_reg: float,
    dropout: float,
    use_batch_norm: bool = False,
    kernel_initializer: str = 'glorot_uniform',
    conv_kernel_size: int = 3,
    conv_strides: int = 1,
    conv_padding: str = 'same',
    use_global_pooling: bool = False,
    **argv
) -> keras.Model:
    inputs = tf.keras.layers.Input(shape=(sequence_length, n_features))
    x = inputs
    current_dim = 3  # Начинаем с 3D (последовательность)
    
    for i in range(disc_layers):
        if disc_layer_types[i] == 'Conv1D':
            # Conv1D всегда работает с 3D
            if current_dim != 3:
                # Преобразуем в 3D
                x = tf.keras.layers.Reshape((sequence_length, disc_layer_sizes[i-1] if i > 0 else n_features))(x)
                current_dim = 3
            
            x = tf.keras.layers.Conv1D(
                filters=disc_layer_sizes[i],
                kernel_size=conv_kernel_size,
                strides=conv_strides,
                padding=conv_padding,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            current_dim = 3
            
        elif disc_layer_types[i] == 'Dense':
            # Если вход 3D, преобразуем в 2D
            if current_dim == 3:
                x = tf.keras.layers.Flatten()(x)
                current_dim = 2
            
            x = tf.keras.layers.Dense(
                disc_layer_sizes[i],
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            current_dim = 2
            
        elif disc_layer_types[i] == 'LSTM':
            # Если вход 2D, преобразуем в 3D
            if current_dim == 2:
                x = tf.keras.layers.Reshape((sequence_length, disc_layer_sizes[i-1] if i > 0 else n_features))(x)
                current_dim = 3
            
            x = tf.keras.layers.LSTM(
                disc_layer_sizes[i],
                activation=activation,
                return_sequences=(i < disc_layers - 1),
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            current_dim = 3 if (i < disc_layers - 1) else 2
            
        elif disc_layer_types[i] == 'GRU':
            if current_dim == 2:
                x = tf.keras.layers.Reshape((sequence_length, disc_layer_sizes[i-1] if i > 0 else n_features))(x)
                current_dim = 3
            
            x = tf.keras.layers.GRU(
                disc_layer_sizes[i],
                activation=activation,
                return_sequences=(i < disc_layers - 1),
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            current_dim = 3 if (i < disc_layers - 1) else 2
        
        # Добавляем BatchNorm и Dropout
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    
    # Финальные слои
    if use_global_pooling:
        if current_dim == 3:
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
        else:
            # Если уже 2D, GlobalAveragePooling1D не нужен
            pass
        current_dim = 2
    else:
        if current_dim == 3:
            x = tf.keras.layers.Flatten()(x)
            current_dim = 2
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

def train_step(
    real_data: np.ndarray,
    generator: keras.Model,
    discriminator: keras.Model,
    gen_optimizer: keras.optimizers.Optimizer,
    disc_optimizer: keras.optimizers.Optimizer,
    batch_size: int,
    disc_steps: int = 1,
    gen_steps: int = 1
) -> Tuple[tf.Tensor, tf.Tensor]:
    # Преобразуем real_data в tf.Tensor
    real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)
    latent_dim = generator.input_shape[1]
    



    # Обучение дискриминатора
    for _ in range(disc_steps):
        noise = tf.random.normal([batch_size, latent_dim])
        generated_data = generator(noise, training=True)
        
        with tf.GradientTape() as disc_tape:
            real_output = discriminator(real_data, training=True)
            fake_output = discriminator(generated_data, training=True)
            real_loss = keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
            fake_loss = keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = tf.reduce_mean(real_loss + fake_loss)
        
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    # Обучение генератора
    for _ in range(gen_steps):
        noise = tf.random.normal([batch_size, latent_dim])
        
        with tf.GradientTape() as gen_tape:
            generated_data = generator(noise, training=True)
            fake_output = discriminator(generated_data, training=True)
            gen_loss = tf.reduce_mean(keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))
        
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    
    #print(f"disc_loss type: {type(disc_loss)}")
    #print(f"gen_loss type: {type(gen_loss)}")


    return disc_loss, gen_loss


def train_gan(
    X_train: np.ndarray,
    X_val: np.ndarray,
    generator: keras.Model,
    discriminator: keras.Model,
    gen_optimizer: keras.optimizers.Optimizer,
    disc_optimizer: keras.optimizers.Optimizer,
    epochs: int,
    batch_size: int,
    disc_steps: int = 1,
    gen_steps: int = 1
) -> Tuple[Dict[str, List[float]], keras.Model, keras.Model]:
    history = {'disc_loss': [], 'gen_loss': [], 'val_score': []}
    best_val_score = -1
    best_generator = None
    best_discriminator = None
    
    for epoch in range(epochs):
        epoch_disc_loss = []
        epoch_gen_loss = []
        
        # Перемешиваем данные
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        
        # Обучение по батчам
        for i in range(0, len(X_train_shuffled), batch_size):
            batch = X_train_shuffled[i:i+batch_size]
            if len(batch) < batch_size:
                continue
            disc_loss, gen_loss = train_step(
                batch,
                generator,
                discriminator,
                gen_optimizer,
                disc_optimizer,
                batch_size,
                disc_steps,
                gen_steps
            )
            # Используем float() вместо .numpy()
            epoch_disc_loss.append(float(disc_loss))
            epoch_gen_loss.append(float(gen_loss))
        
        # Оценка на валидации
        val_probs = discriminator.predict(X_val)
        val_score = np.mean(val_probs)
        
        # Сохраняем лучшую модель
        if val_score > best_val_score:
            best_val_score = val_score
            best_generator = keras.models.clone_model(generator)
            best_discriminator = keras.models.clone_model(discriminator)
            best_generator.set_weights(generator.get_weights())
            best_discriminator.set_weights(discriminator.get_weights())
        
        # Сохраняем историю
        history['disc_loss'].append(np.mean(epoch_disc_loss))
        history['gen_loss'].append(np.mean(epoch_gen_loss))
        history['val_score'].append(val_score)
        
        print(f"Epoch {epoch+1}/{epochs} | Disc Loss: {np.mean(epoch_disc_loss):.4f} | Gen Loss: {np.mean(epoch_gen_loss):.4f} | Val Score: {val_score:.4f}")
    
    return history, best_generator, best_discriminator

def grid_search(
    X: np.ndarray,
    param_grid: Dict[str, List],
    sequence_length: int,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 1
) -> Tuple[Dict, keras.Model, keras.Model, pd.DataFrame]:
    """
    Выполняет grid search для подбора гиперпараметров GAN.
    
    Args:
        X: Исходные данные (n_samples, n_features)
        param_grid: Сетка параметров
        sequence_length: Длина окна
        test_size: Размер тестовой выборки
        random_state: Сид для воспроизводимости
        n_splits: Количество сплитов для кросс-валидации
    
    Returns:
        Tuple[Dict, keras.Model, keras.Model, pd.DataFrame]: Лучшие параметры, лучшая модель, история обучения
    """
    n_samples, n_features = X.shape
    X_processed = preprocess_data(X, sequence_length)
    
    # Разбиваем данные
    X_train, X_val = train_test_split(X_processed, test_size=test_size, random_state=random_state)
    
    # Генерируем все комбинации параметров
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    all_combinations = list(itertools.product(*values))
    random.shuffle(all_combinations)
    all_combinations = all_combinations[:100]
    results = []
    
    for i, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))
        params["gen_layers"] = len(params['gen_layer_types'])
        params["disc_layers"] = len(params['disc_layer_types'])
        params["gen_layer_sizes"] = []
        params["disc_layer_sizes"] = []

        xx = 32 
        for _ in range(params["gen_layers"]):
            params["gen_layer_sizes"].append(xx)
            xx*=2


        xx = 32
        for _ in range(params["disc_layers"]):
            params["disc_layer_sizes"].append(xx)
            xx*=2

        print(f"\nПопытка {i+1}/{len(all_combinations)}: {params}")



        try:
            # Создаем модели
            generator = create_generator(
                latent_dim=params['latent_dim'],
                sequence_length=sequence_length,
                n_features=n_features,
                gen_layers=params["gen_layers"],
                gen_layer_types=params['gen_layer_types'],
                gen_layer_sizes=params['gen_layer_sizes'],
                activation=params['activation'],
                l2_reg=params['l2_reg'],
                dropout=params['dropout'],
                use_batch_norm=params['use_batch_norm'],
                kernel_initializer=params['kernel_initializer'],
                use_skip_connections=params['use_skip_connections']
            )
            
            discriminator = create_discriminator(
                sequence_length=sequence_length,
                n_features=n_features,
                disc_layers=len(params['disc_layer_types']),
                disc_layer_types=params['disc_layer_types'],
                disc_layer_sizes=params['disc_layer_sizes'],
                activation=params['activation'],
                l2_reg=params['l2_reg'],
                dropout=params['dropout'],
                use_batch_norm=params['use_batch_norm'],
                kernel_initializer=params['kernel_initializer'],
                conv_kernel_size=params['conv_kernel_size'],
                conv_strides=params['conv_strides'],
                conv_padding=params['conv_padding'],
                use_global_pooling=params['use_global_pooling']
            )
            
            # Создаем оптимизаторы
            gen_optimizer = keras.optimizers.Adam(learning_rate=params['lr'])
            disc_optimizer = keras.optimizers.Adam(learning_rate=params['lr'])
            
            # Обучаем модель
            history, best_gen, best_disc = train_gan(
                X_train,
                X_val,
                generator,
                discriminator,
                gen_optimizer,
                disc_optimizer,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                disc_steps=params['disc_steps'],
                gen_steps=params['gen_steps']
            )
            
            # Оценка качества
            val_probs = best_disc.predict(X_val)
            val_score = np.mean(val_probs)
            
            # Сохраняем результаты
            result = {
                'params': params,
                'val_score': val_score,
                'best_disc_loss': min(history['disc_loss']),
                'best_gen_loss': min(history['gen_loss']),
                'best_val_score': val_score
            }
            results.append(result)
            print(f"Успех! Val Score: {val_score:.4f}")
        
        except Exception as e:
            print(f"Ошибка при попытке {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Находим лучшие параметры
    if not results:
        raise ValueError("Не удалось создать ни одну модель")
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['val_score'].idxmax()
    best_params = results_df.loc[best_idx, 'params']
    #best_generator = create_generator(**best_params, sequence_length=sequence_length, n_features=n_features)
    #best_discriminator = create_discriminator(**best_params, sequence_length=sequence_length, n_features=n_features)
    
    # Возвращаем лучшие параметры, модели и историю
    return best_params,  results_df

def detect_anomalies(
    data: np.ndarray,
    generator: keras.Model,
    discriminator: keras.Model,
    sequence_length: int,
    stride: int = 1
) -> np.ndarray:
    """
    Обнаруживает аномалии в данных.
    
    Args:
        data: Исходные данные (n_samples, n_features)
        generator: Обученный генератор
        discriminator: Обученный дискриминатор
        sequence_length: Длина окна
        stride: Шаг сдвига окна
    
    Returns:
        np.ndarray: Массив оценок аномалий (n_samples, 1)
    """
    n_samples, n_features = data.shape
    X_processed = preprocess_data(data, sequence_length, stride)
    
    # Получаем вероятности дискриминатора для каждого окна
    window_probs = discriminator.predict(X_processed)
    
    # Преобразуем оценки окон в оценки для каждого временного шага
    anomaly_scores = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    
    for i in range(len(X_processed)):
        start = i * stride
        end = start + sequence_length
        anomaly_scores[start:end] += window_probs[i]
        counts[start:end] += 1
    
    # Нормализуем оценки
    anomaly_scores = anomaly_scores / np.maximum(counts, 1)
    return anomaly_scores.reshape(-1, 1)
