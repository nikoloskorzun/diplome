# Используем официальный образ с предустановленным Jupyter
FROM jupyter/base-notebook:python-3.11

# Устанавливаем зависимости из requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Рабочая директория внутри контейнера
WORKDIR /home/jovyan/work

# Пользователь jovyan (UID 1000) используется по умолчанию в образе jupyter