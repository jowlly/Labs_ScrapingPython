import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
from scipy import stats


class CurrencyAnalytics:
    def __init__(self):
        self.df = None
        self.processed_df = None
    
    def load_data(self, file_path):
        """Загрузка данных из CSV файла"""
        try:
            self.df = pd.read_csv(file_path)
            return True, f"Данные загружены. Записей: {len(self.df)}"
        except Exception as e:
            return False, f"Ошибка при загрузке данных: {str(e)}"
    
    def prepare_data(self, start_date, end_date):
        """Подготовка и обработка данных"""
        try:
            self.processed_df = self.filter_by_date(self.df, start_date, end_date)
            self.processed_df.columns = ['date', 'value']
            
            self.processed_df['value'] = self.processed_df['value'].fillna(self.processed_df['value'].mean())
            
            median_value = self.processed_df['value'].median()
            mean_value = self.processed_df['value'].mean()
            self.processed_df['deviation_median'] = self.processed_df['value'] - median_value
            self.processed_df['deviation_mean'] = self.processed_df['value'] - mean_value
            
            self.processed_df['date'] = pd.to_datetime(self.processed_df['date'])
            
            return True, f"Данные подготовлены. Записей: {len(self.processed_df)}"
        except Exception as e:
            return False, f"Ошибка при подготовке данных: {str(e)}"
    
    def filter_by_deviation(self, dataframe, threshold):
        """Фильтрация по значению отклонению от курса"""
        return dataframe[dataframe['deviation_mean'] >= threshold]
    
    def filter_by_date(self, dataframe, start_date, end_date):
        """Фильтрация по датам"""
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        mask = (dataframe['date'] >= start_date) & (dataframe['date'] <= end_date)
        return dataframe.loc[mask]
    
    def plot_month_data(self, month):
        """Построение изменения курса за месяц и возврат base64 изображения"""
        try:
            monthly_data = self.processed_df[self.processed_df['date'].dt.to_period('M') == month]
            if monthly_data.empty:
                return False, "Нет данных за указанный месяц"
            
            median_val = monthly_data['value'].median()
            mean_val = monthly_data['value'].mean()
            
            plt.figure(figsize=(12, 6))
            plt.plot(monthly_data['date'], monthly_data['value'], label='Курс')
            plt.axhline(median_val, color='red', linestyle='--', label=f'Медиана: {median_val:.2f}')
            plt.axhline(mean_val, color='green', linestyle='-.', label=f'Среднее: {mean_val:.2f}')
            plt.xlabel('Дата')
            plt.ylabel('Значение курса')
            plt.title(f'Курс за {month} с выделением медианы и среднего')
            plt.legend()
            plt.grid(True)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
            return True, img_base64
        except Exception as e:
            return False, f"Ошибка при построении графика: {str(e)}"
    
    def plot_full_period(self):
        """Построение графика за весь период"""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(self.processed_df['date'], self.processed_df['value'], label='Курс')
            plt.xlabel('Дата')
            plt.ylabel('Значение курса')
            plt.title('Изменение курса за весь период')
            plt.legend()
            plt.grid(True)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
            return True, img_base64
        except Exception as e:
            return False, f"Ошибка при построении графика: {str(e)}"
    
    def show_boxplot(self):
        """Построение Boxplot"""
        try:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.processed_df[['value', 'deviation_median', 'deviation_mean']])
            plt.title('Распределение данных и выбросы')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
            return True, img_base64
        except Exception as e:
            return False, f"Ошибка при построении boxplot: {str(e)}"
    
    def get_statistics(self):
        """Получение статистики по данным"""
        try:
            stats = self.processed_df[['value', 'deviation_median', 'deviation_mean']].describe()
            return True, f"Статистика по данным:\n{stats}"
        except Exception as e:
            return False, f"Ошибка при вычислении статистики: {str(e)}"
    
    def filter_data_by_deviation(self, threshold):
        """Фильтрация данных по отклонению"""
        try:
            filtered_df = self.filter_by_deviation(self.processed_df, threshold)
            return True, f"Отфильтровано по отклонению ≥ {threshold}: {len(filtered_df)} записей"
        except Exception as e:
            return False, f"Ошибка при фильтрации: {str(e)}"
    
    def filter_data_by_date_range(self, start_date, end_date):
        """Фильтрация данных по диапазону дат"""
        try:
            filtered_df = self.filter_by_date(self.processed_df, start_date, end_date)
            return True, f"Отфильтровано по дате с {start_date} по {end_date}: {len(filtered_df)} записей"
        except Exception as e:
            return False, f"Ошибка при фильтрации по дате: {str(e)}"
    
    def analyze_moving_window(self, window_size=12):
        """Анализ методом скользящего окна на 12 шагов"""
        try:
            if self.processed_df is None or len(self.processed_df) == 0:
                return False, "Нет данных для анализа"
            
            values = self.processed_df['value'].values
            dates = self.processed_df['date'].values
            
            moving_means = []
            moving_stds = []
            moving_dates = []
            
            for i in range(window_size, len(values)):
                window = values[i-window_size:i]
                moving_means.append(np.mean(window))
                moving_stds.append(np.std(window))
                moving_dates.append(dates[i])
            
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            
            # График 1: Исходные данные и скользящее среднее
            axes[0].plot(dates, values, label='Исходные данные', alpha=0.7)
            axes[0].plot(moving_dates, moving_means, label=f'Скользящее среднее (окно={window_size})', color='red', linewidth=2)
            axes[0].set_title('Исходные данные и скользящее среднее')
            axes[0].set_xlabel('Дата')
            axes[0].set_ylabel('Значение')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # График 2: Скользящее стандартное отклонение
            axes[1].plot(moving_dates, moving_stds, color='green', linewidth=2)
            axes[1].set_title(f'Скользящее стандартное отклонение (окно={window_size})')
            axes[1].set_xlabel('Дата')
            axes[1].set_ylabel('Стандартное отклонение')
            axes[1].grid(True, alpha=0.3)
            
            # График 3: Отклонение от скользящего среднего
            deviations = values[window_size:] - moving_means
            axes[2].plot(moving_dates, deviations, color='purple', linewidth=1.5)
            axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[2].fill_between(moving_dates, 0, deviations, where=deviations>0, 
                                color='green', alpha=0.3, label='Выше среднего')
            axes[2].fill_between(moving_dates, 0, deviations, where=deviations<=0, 
                                color='red', alpha=0.3, label='Ниже среднего')
            axes[2].set_title('Отклонение от скользящего среднего')
            axes[2].set_xlabel('Дата')
            axes[2].set_ylabel('Отклонение')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
            # Анализ автокорреляции
            correlation_analysis = self._analyze_autocorrelation(values, window_size)
            
            return True, {
                'plot': img_base64,
                'analysis': {
                    'window_size': window_size,
                    'total_windows': len(moving_means),
                    'avg_moving_mean': np.mean(moving_means),
                    'avg_moving_std': np.mean(moving_stds),
                    'max_deviation': np.max(np.abs(deviations)),
                    'autocorrelation': correlation_analysis
                }
            }
            
        except Exception as e:
            return False, f"Ошибка при анализе скользящего окна: {str(e)}"
    
    def _analyze_autocorrelation(self, data, max_lag=12):
        """Анализ автокорреляции с выводами"""
        try:
            from statsmodels.tsa.stattools import acf, pacf
            
            # Вычисление автокорреляции
            autocorr_values = acf(data, nlags=max_lag, fft=True)
            partial_autocorr_values = pacf(data, nlags=max_lag)
            
            # Анализ результатов
            analysis = {
                'significant_lags': [],
                'seasonality_detected': False,
                'trend_strength': 'слабый',
                'analysis_summary': ''
            }
            
            # Порог значимости (95% доверительный интервал)
            significance_threshold = 1.96 / np.sqrt(len(data))
            
            # Проверка значимых лагов
            for lag in range(1, max_lag + 1):
                if abs(autocorr_values[lag]) > significance_threshold:
                    analysis['significant_lags'].append(lag)
            
            # Проверка на сезонность (повторение на лаге 12 или его кратных)
            seasonal_lags = [lag for lag in analysis['significant_lags'] if lag % 12 == 0 or lag == 12]
            if seasonal_lags:
                analysis['seasonality_detected'] = True
            
            # Оценка силы тренда по автокорреляции первого порядка
            if autocorr_values[1] > 0.7:
                analysis['trend_strength'] = 'сильный'
            elif autocorr_values[1] > 0.3:
                analysis['trend_strength'] = 'умеренный'
            
            # Формирование выводов
            summary = "ВЫВОДЫ ПО АВТОКОРРЕЛЯЦИИ:\n"
            summary += f"1. Общее количество наблюдений: {len(data)}\n"
            summary += f"2. Порог значимости (95%): ±{significance_threshold:.3f}\n\n"
            summary += "3. Значимые лаги автокорреляции:\n"
            if analysis['significant_lags']:
                for lag in analysis['significant_lags']:
                    summary += f"   - Лаг {lag}: {autocorr_values[lag]:.3f}\n"
            else:
                summary += "   - Нет значимых лагов\n"
            
            summary += f"\n4. Сезонность: {'ОБНАРУЖЕНА' if analysis['seasonality_detected'] else 'не обнаружена'}\n"
            summary += f"5. Сила тренда: {analysis['trend_strength']}\n\n"
            
            # Интерпретация результатов
            if analysis['seasonality_detected']:
                summary += "ИНТЕРПРЕТАЦИЯ:\n"
                summary += "• Обнаружены сезонные паттерны (12-месячный цикл)\n"
                summary += "• Рекомендуется использовать модели с сезонной компонентой (SARIMA, Holt-Winters)\n"
            elif len(analysis['significant_lags']) > 0:
                summary += "ИНТЕРПРЕТАЦИЯ:\n"
                summary += "• Обнаружена краткосрочная зависимость между наблюдениями\n"
                summary += "• Возможно присутствие тренда\n"
            else:
                summary += "ИНТЕРПРЕТАЦИЯ:\n"
                summary += "• Данные близки к случайным (белый шум)\n"
                summary += "• Прогнозирование может быть затруднено\n"
            
            # Добавление матрицы автокорреляции в анализ
            analysis['autocorr_matrix'] = autocorr_values.tolist()
            analysis['partial_autocorr_matrix'] = partial_autocorr_values.tolist()
            analysis['analysis_summary'] = summary
            
            return analysis
            
        except Exception as e:
            return f"Ошибка анализа автокорреляции: {str(e)}"


def filter_by_deviation(dataframe, threshold):
    """Фильтрация по значению отклонению от курса"""
    return dataframe[dataframe['deviation_mean'] >= threshold]

def filter_by_date(dataframe, start_date, end_date):
    """Фильтрация по датам"""
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    mask = (dataframe['date'] >= start_date) & (dataframe['date'] <= end_date)
    return dataframe.loc[mask]

def plot_month_data(dataframe, month):
    """Построение изменения курса за месяц (для обратной совместимости)"""
    analytics = CurrencyAnalytics()
    analytics.processed_df = dataframe
    success, result = analytics.plot_month_data(month)
    if success:
        fig = plt.figure()
        return fig
    else:
        return None

if __name__ == "__main__":
    df = pd.read_csv('dataset.csv')
    df = filter_by_date(df, '2000-01-01', '2025-10-05')
    df.columns = ['date', 'value']

    print("Пропуски в данных:")
    print(df.isnull().sum())
    df['value'] = df['value'].fillna(df['value'].mean())

    median_value = df['value'].median()
    mean_value = df['value'].mean()
    df['deviation_median'] = df['value'] - median_value
    df['deviation_mean'] = df['value'] - mean_value

    print("\nСтатистика по числовым колонкам:")
    print(df[['value', 'deviation_median', 'deviation_mean']].describe())

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[['value', 'deviation_median', 'deviation_mean']])
    plt.title('Распределение данных и выбросы')
    plt.show()

    df['date'] = pd.to_datetime(df['date'])
    monthly_avg = df.groupby(pd.Grouper(key='date', freq='M'))['value'].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['value'], label='Курс')
    plt.xlabel('Дата')
    plt.ylabel('Значение курса')
    plt.title('Изменение курса за весь период')
    plt.legend()
    plt.grid(True)
    plt.show()

    filtered_dev = filter_by_deviation(df, 100)
    print(f"\nОтфильтровано по отклонению ≥ 100: {len(filtered_dev)} записей")
    filtered_date = filter_by_date(df, '1995-01-01', '1995-01-31')
    print(f"Отфильтровано по дате: {len(filtered_date)} записей")
    plot_month_data(df, '2025-01')