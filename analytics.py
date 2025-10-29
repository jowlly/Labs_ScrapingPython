import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def filter_by_deviation(dataframe, threshold):
    """Фильтрация по значению отклонению от курса"""
    return dataframe[dataframe['deviation_mean'] >= threshold]

def filter_by_date(dataframe, start_date, end_date):
    """Фильтрация по датам"""
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    mask = (dataframe['date'] >= start_date) & (dataframe['date'] <= end_date)
    return dataframe.loc[mask]

def plot_month_data(dataframe, month):
    """Построение изменения курса за месяц"""
    monthly_data = dataframe[dataframe['date'].dt.to_period('M') == month]
    if monthly_data.empty:
        print("Нет данных за указанный месяц")
        return
    
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
    plt.show()

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