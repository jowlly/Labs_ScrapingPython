import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64


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
    #1
    #В 1999 резкое изменение, поэтому берём с 2000
    df = pd.read_csv('dataset.csv')
    df = filter_by_date(df, '2000-01-01', '2025-10-05')
    df.columns = ['date', 'value']

    #3
    print("Пропуски в данных:")
    print(df.isnull().sum())
    df['value'] = df['value'].fillna(df['value'].mean())

    #4
    median_value = df['value'].median()
    mean_value = df['value'].mean()
    df['deviation_median'] = df['value'] - median_value
    df['deviation_mean'] = df['value'] - mean_value

    #5
    print("\nСтатистика по числовым колонкам:")
    print(df[['value', 'deviation_median', 'deviation_mean']].describe())

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[['value', 'deviation_median', 'deviation_mean']])
    plt.title('Распределение данных и выбросы')
    plt.show()

    #8
    df['date'] = pd.to_datetime(df['date'])
    monthly_avg = df.groupby(pd.Grouper(key='date', freq='M'))['value'].mean().reset_index()

    #9
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['value'], label='Курс')
    plt.xlabel('Дата')
    plt.ylabel('Значение курса')
    plt.title('Изменение курса за весь период')
    plt.legend()
    plt.grid(True)
    plt.show()

    #6
    filtered_dev = filter_by_deviation(df, 100)
    print(f"\nОтфильтровано по отклонению ≥ 100: {len(filtered_dev)} записей")
    #7
    filtered_date = filter_by_date(df, '1995-01-01', '1995-01-31')
    print(f"Отфильтровано по дате: {len(filtered_date)} записей")
    #10
    plot_month_data(df, '2025-01')