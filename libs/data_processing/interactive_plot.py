import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from .dataset import Dataset
def plot_interactive(dataset: Dataset):
    """
    Интерактивный график с выбором столбцов и цветовой маркировкой по целевой переменной
    
    Параметры:
    dataset (Dataset): Исходный датасет
    """
    target_column = dataset.target_features[0]

    # Проверка наличия целевого столбца
    if target_column not in dataset.columns:
        raise ValueError(f"Столбец {target_column} не найден в датасете")

    # Подготовка данных
    features = dataset.columns.drop(target_column).tolist()
    colors = dataset[target_column].map({0: 'green', 1: 'red'})

    # Создание виджетов
    feature_selector = widgets.SelectMultiple(
        options=features,
        description='Признаки:',
        rows=10,
        disabled=False
    )

    # Создание контейнера для графика
    output = widgets.Output()

    # Функция обновления графика
    def update_plot(selected_features):
        with output:
            output.clear_output(wait=True)
            
            plt.figure(figsize=(12, 6))
            x = range(1, len(dataset)+1)
            
            # Маркеры для разных признаков
            markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'X', 'D']
            
            for i, feature in enumerate(selected_features):
                plt.scatter(
                    x, 
                    dataset[feature], 
                    c=colors,
                    marker=markers[i % len(markers)],
                    alpha=0.7,
                    label=feature
                )
            
            plt.title(f'Интерактивный график признаков (Целевая переменная: {target_column})')
            plt.xlabel('Номер наблюдения')
            plt.ylabel('Значение признака')
            plt.grid(True)
            
            if selected_features:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.show()

    # Связывание виджетов с функцией
    widgets.interactive_output(update_plot, {'selected_features': feature_selector})
    
    # Отображение элементов
    display(widgets.VBox([feature_selector, output]))