import pandas as pd
import requests
from datetime import datetime
from enum import Enum
import os
import glob


class SourceType(Enum):
    ALL=0,
    WEEKS=1,
    YEARS=2,
    SPLITTED=3

def scrape_val(val: str = 'USD', fromyear: int = 2025, years: int = 1, 
            frommonth: int = 9, fromday: int = 1):
    """
    Скрапинг данных о валюте с сайта ЦБ РФ
    """
    url = 'https://www.cbr-xml-daily.ru/archive/'

    result=[]
    for y in range(years):
        for m in range(frommonth,13):
            for d in range(fromday,32):
                cur_date = f"{fromyear+y}/{m if m>9 else '0'+str(m)}/{d if d>9 else '0'+str(d)}"
                if cur_date > datetime.now().date().strftime("%Y/%m/%d"):
                    break
                cur_url = url +cur_date +"/daily_json.js"

                print("Скрапинг страницы: ", cur_url)

                json_response = requests.get(cur_url).json()
                if("error" in json_response.keys()):
                    print("Не найдено")
                    continue
                print(json_response["Valute"][val])
                result.append([str(json_response["Date"]).split("T")[0],json_response["Valute"][val]["Value"]])

    df = pd.DataFrame(result, columns = ["date", "value"])
    return df

def save_to_csv(df, filename: str = 'dataset.csv'):
    df.to_csv(filename, index=False)

def split_to_x_y(input_file='dataset.csv'):
    df = pd.read_csv(input_file)
    df[['date']].to_csv('X.csv', index=False)
    df[['value']].to_csv('Y.csv', index=False)
    print("Файл разделен на X.csv и Y.csv")

def split_csv_by_years(input_file='dataset.csv', output_dir='yearly_data'):
    """Разделяет CSV файл по годам"""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    
    for year, group in df.groupby(df['date'].dt.year):
        start_date = group['date'].min().strftime('%Y%m%d')
        end_date = group['date'].max().strftime('%Y%m%d')
        filename = f"{start_date}_{end_date}.csv"
        group.to_csv(os.path.join(output_dir, filename), index=False)
    
    print(f"Файлы по годам сохранены в директорию {output_dir}")

def split_csv_by_weeks(input_file='dataset.csv', output_dir='weekly_data'):
    """Разделяет CSV файл по неделям"""
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df['year_week'] = df['date'].dt.strftime('%Y-%U')
    
    for week, group in df.groupby('year_week'):
        start_date = group['date'].min().strftime('%Y%m%d')
        end_date = group['date'].max().strftime('%Y%m%d')
        filename = f"{start_date}_{end_date}.csv"
        group.drop('year_week', axis=1).to_csv(os.path.join(output_dir, filename), index=False)
    
    print(f"Файлы по неделям сохранены в директорию {output_dir}")

class DataReader:
    """Класс для чтения данных по дате из разных источников"""
    
    def __init__(self, source_type=SourceType.ALL, source_path=None):
        self.source_type = source_type
        self.source_path = source_path or self._get_default_source()
        self.data = self._load_data()
    
    def _get_default_source(self):
        sources = ['dataset.csv', 'yearly_data', 'weekly_data', '.']
        return sources[self.source_type.value[0]]
    
    def _load_data(self):
        if self.source_type == SourceType.ALL:
            return pd.read_csv(self.source_path)
        elif self.source_type == SourceType.WEEKS or self.source_type == SourceType.YEARS:
            files = glob.glob(os.path.join(self.source_path, "*.csv"))
            dfs = []
            for file in sorted(files):
                dfs.append(pd.read_csv(file))
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['date', 'value'])
        elif self.source_type == SourceType.SPLITTED:
            try:
                df_x = pd.read_csv('X.csv')
                df_y = pd.read_csv('Y.csv')
                return pd.concat([df_x, df_y], axis=1)
            except FileNotFoundError:
                return pd.DataFrame(columns=['date', 'value'])
        
        return pd.DataFrame(columns=['date', 'value'])
    
    def get_data_by_date(self, date):
        """Возвращает данные для указанной даты"""
        if self.data is None or self.data.empty:
            return None
            
        date_str = date.strftime('%Y-%m-%d')
        result = self.data[self.data['date'] == date_str]
        
        if len(result) == 0:
            return None
        
        if self.source_type == SourceType.SPLITTED:
            return float(result['value'].iloc[0])
        else:
            return float(result.iloc[0]['value'])

def get_data_by_date(date):
    reader = DataReader(SourceType.ALL)
    return reader.get_data_by_date(date)

def get_data_by_date_weekly(date):
    reader = DataReader(SourceType.WEEKS)
    return reader.get_data_by_date(date)

def get_data_by_date_yearly(date):
    reader = DataReader(SourceType.YEARS)
    return reader.get_data_by_date(date)

def get_data_by_date_splitted(date):
    reader = DataReader(SourceType.SPLITTED)
    return reader.get_data_by_date(date)

class DataIterator:
    """Итератор для последовательного чтения данных"""
    
    def __init__(self, source_type=0, source_path=None):
        self.reader = DataReader(source_type, source_path)
        self.current_index = 0
        
        if self.reader.data is not None and not self.reader.data.empty:
            self.sorted_data = self.reader.data.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        else:
            self.sorted_data = pd.DataFrame(columns=['date', 'value'])
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= len(self.sorted_data):
            raise StopIteration
        
        row = self.sorted_data.iloc[self.current_index]
        self.current_index += 1
        
        date = datetime.strptime(row['date'], '%Y-%m-%d')
        value = float(row['value']) if 'value' in row else float(row[1])
        
        return (date, value)

if __name__ == '__main__':

    #Лабораторная работа №1
    # df = scrape_val()
    # save_to_csv(df)

    # Разделение на X и Y
    split_to_x_y('dataset.csv')
    
    # Разделение по годам
    split_csv_by_years('dataset.csv', 'yearly_data')
    
    # Разделение по неделям
    split_csv_by_weeks('dataset.csv', 'weekly_data')
    
    print("\Тест итератора (5 записей):")
    iterator = DataIterator(SourceType.ALL, 'dataset.csv')
    for i, (date, value) in enumerate(iterator):
        print(f"{date.strftime('%Y-%m-%d')}: {value}")
        if i >= 4:
            break

    print("\Поиска по дате:")
    test_date = datetime(2025, 9, 2)
    result = get_data_by_date(test_date)
    print(f"Данные за {test_date.strftime('%Y-%m-%d')}: {result}")
