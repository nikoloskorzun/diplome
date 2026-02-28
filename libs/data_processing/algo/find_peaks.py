import numpy as np
from scipy.signal import find_peaks, detrend
def detect_multivariate_period(data, min_period=5, oscillation_threshold=0.3, max_period_ratio=0.5):
    """
    Обнаруживает общий период, предварительно отфильтровывая непериодические ряды.
    
    Параметры:
    ----------
    data : np.ndarray
        Входные данные формы (время, каналы).
    min_period : int
        Минимально допустимый период.
    oscillation_threshold : float
        Порог силы автокорреляции для того, чтобы считать ряд 'колеблющимся' (0.0 - 1.0).
        Каналы с пиком ACF ниже этого значения будут игнорироваться.
    max_period_ratio : float
        Максимальный период как доля от длины ряда.
        
    Возвращает:
    -----------
    dict : {
        'period': int or None,      # Найденный общий период
        'active_channels': list,    # Индексы каналов, которые были признаны периодическими
        'scores': list              # Оценки силы периодичности для активных каналов
    }
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        
    n_time, n_channels = data.shape
    
    if n_time < 10:
        return {'period': None, 'active_channels': [], 'scores': []}

    max_lag = int(n_time * max_period_ratio)
    max_lag = max(min_period + 1, max_lag)
    
    # Хранилище для ACF только хороших каналов
    valid_acfs = []
    active_channels = []
    channel_scores = []

    for i in range(n_channels):
        series = data[:, i]
        
        # 1. Предобработка: Удаление тренда и нормализация
        # detrend(type='linear') убирает линейный тренд, который портит ACF
        series_dt = detrend(series, type='linear')
        
        mean = np.mean(series_dt)
        std = np.std(series_dt)
        
        if std < 1e-9: # Если сигнал константный (после удаления тренда)
            continue
            
        series_norm = (series_dt - mean) / std
        
        # 2. Расчет автокорреляции
        correlation = np.correlate(series_norm, series_norm, mode='full')
        acf = correlation[n_time-1:]
        
        # Нормализация ACF (unbiased estimator для корректности на больших лагах)
        denominator = np.arange(n_time, 0, -1)
        acf_norm = acf / denominator
        
        # Обрезаем до max_lag
        if len(acf_norm) > max_lag:
            acf_norm = acf_norm[:max_lag]
        
        # 3. Поиск локального пика (кандидата в периоды)
        # Ищем пики только после min_period
        search_slice = acf_norm[min_period:]
        
        if len(search_slice) < 2:
            continue
            
        peaks, properties = find_peaks(search_slice, height=oscillation_threshold)
        
        if len(peaks) > 0:
            # Берем самый высокий пик как индикатор силы периодичности этого канала
            best_peak_idx_in_slice = peaks[np.argmax(properties['peak_heights'])]
            best_peak_height = properties['peak_heights'][np.argmax(properties['peak_heights'])]
            best_lag = best_peak_idx_in_slice + min_period
            
            # 4. Фильтрация: если пик достаточно высокий, сохраняем канал
            if best_peak_height >= oscillation_threshold:
                valid_acfs.append(acf_norm)
                active_channels.append(i)
                channel_scores.append(float(best_peak_height))

    # 5. Если нет ни одного периодического канала
    if len(valid_acfs) == 0:
        return {'period': None, 'active_channels': [], 'scores': []}

    # 6. Агрегация ACF только от отобранных каналов
    aggregated_acf = np.sum(valid_acfs, axis=0) / len(valid_acfs)
    
    # 7. Финальный поиск общего периода на агрегированной ACF
    # Здесь можно снизить порог, так как шум уже усреднился, 
    # но лучше искать просто самый высокий пик после min_period
    final_peaks, final_props = find_peaks(aggregated_acf[min_period:], height=0.1)
    
    if len(final_peaks) == 0:
        # Если вдруг после суммирования пики стали слишком низкими (редко, но бывает)
        # Возьмем просто аргмакс
        final_period_idx = np.argmax(aggregated_acf[min_period:]) + min_period
        # Проверка на адекватность (если максимум на самом краю - скорее всего шума нет)
        if final_period_idx > max_lag - 2: 
            return {'period': None, 'active_channels': active_channels, 'scores': channel_scores}
    else:
        # Берем самый высокий пик среди найденных
        best_final_idx_in_slice = final_peaks[np.argmax(final_props['peak_heights'])]
        final_period_idx = best_final_idx_in_slice + min_period

    return {
        'period': int(final_period_idx),
        'active_channels': active_channels,
        'scores': channel_scores
    }