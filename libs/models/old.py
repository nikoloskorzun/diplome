
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
    """
    Создает генератор для GAN.
    
    Args:
        latent_dim: Размерность шума
        sequence_length: Длина последовательности на выходе
        n_features: Количество признаков
        layers: Количество слоев
        layer_types: Типы слоев ('Dense', 'LSTM', 'GRU', 'Conv1DTranspose')
        layer_sizes: Размеры слоев
        activation: Активация
        l2_reg: Коэффициент L2-регуляризации
        dropout: Вероятность дропаута
        use_batch_norm: Использовать batch normalization
        kernel_initializer: Инициализатор ядер
        use_skip_connections: Использовать skip connections
    
    Returns:
        keras.Model: Скомпилированный генератор
    """
    inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = inputs
    skip_connections = []
    
    for i in range(gen_layers):
        if gen_layer_types[i] == 'Dense':
            x = tf.keras.layers.Dense(
                gen_layer_sizes[i],
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
        elif gen_layer_types[i] == 'LSTM':
            x = tf.keras.layers.LSTM(
                gen_layer_sizes[i],
                activation=activation,
                return_sequences=(i < gen_layers - 1),
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
        elif gen_layer_types[i] == 'GRU':
            x = tf.keras.layers.GRU(
                gen_layer_sizes[i],
                activation=activation,
                return_sequences=(i < gen_layers - 1),
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
        elif gen_layer_types[i] == 'Conv1DTranspose':
            x = tf.keras.layers.Conv1DTranspose(
                filters=gen_layer_sizes[i],
                kernel_size=3,
                strides=1,
                padding='same',
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
        
        if use_skip_connections and i > 0:
            skip_connections.append(x)
            if len(skip_connections) > 1:
                x = tf.keras.layers.Add()([x, skip_connections[-2]])
    
    # Преобразуем в последовательность
    if gen_layer_types[-1] == 'Conv1DTranspose':
        # Добавляем последний Conv1DTranspose для точного размера
        x = tf.keras.layers.Conv1DTranspose(
            filters=n_features,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='tanh'
        )(x)
    else:
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
    """
    Создает дискриминатор для GAN.
    
    Args:
        sequence_length: Длина последовательности на входе
        n_features: Количество признаков
        layers: Количество слоев
        layer_types: Типы слоев ('Dense', 'LSTM', 'GRU', 'Conv1D')
        layer_sizes: Размеры слоев
        activation: Активация
        l2_reg: Коэффициент L2-регуляризации
        dropout: Вероятность дропаута
        use_batch_norm: Использовать batch normalization
        kernel_initializer: Инициализатор ядер
        conv_kernel_size: Размер ядра для Conv1D
        conv_strides: Страйды для Conv1D
        conv_padding: Padding для Conv1D
        use_global_pooling: Использовать глобальное пулингирование вместо Flatten
    
    Returns:
        keras.Model: Скомпилированный дискриминатор
    """
    inputs = tf.keras.layers.Input(shape=(sequence_length, n_features))
    x = inputs
    
    for i in range(disc_layers):
        if disc_layer_types[i] == 'Conv1D':
            x = tf.keras.layers.Conv1D(
                filters=disc_layer_sizes[i],
                kernel_size=conv_kernel_size,
                strides=conv_strides,
                padding=conv_padding,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            if use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            if dropout > 0:
                x =tf.keras.layers.Dropout(dropout)(x)
        elif disc_layer_types[i] == 'LSTM':
            x = tf.keras.layers.LSTM(
                disc_layer_sizes[i],
                activation=activation,
                return_sequences=(i < disc_layers - 1),
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            if use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            if dropout > 0:
                x = tf.keras.layers.Dropout(dropout)(x)
        elif disc_layer_types[i] == 'GRU':
            x = tf.keras.layers.GRU(
                disc_layer_sizes[i],
                activation=activation,
                return_sequences=(i < disc_layers - 1),
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            if use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            if dropout > 0:
                x = tf.keras.layers.Dropout(dropout)(x)
        elif disc_layer_types[i] == 'Dense':
            x = tf.keras.layers.Dense(
                disc_layer_sizes[i],
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
                kernel_initializer=kernel_initializer
            )(x)
            if use_batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            if dropout > 0:
                x = tf.keras.layers.Dropout(dropout)(x)
    
    # Финальные слои
    if use_global_pooling:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    else:
        x = tf.keras.layers.Flatten()(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inputs, outputs)



"""