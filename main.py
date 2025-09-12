import pandas as pd
import requests
from datetime import datetime

def scrape_val(val: str = 'USD', fromyear: int = 2025, years: int = 1, 
            frommonth: int = 9, fromday: int = 1):
    """
    Скрапинг данных о валюте с сайта ЦБ РФ
    
    Args:
        val: Код валюты
        foryear: Начальный год
        years: Количество лет
        frommonth: Начальный месяц
        fromday: Начальный день
    
    Returns:
        DataFrame с колонками ['date', 'value']
    """
    url = 'https://www.cbr-xml-daily.ru/archive/'

    result=[]
    for y in range(years):
        for m in range(frommonth,13):
            for d in range(fromday,32):
                cur_date = f"{fromyear+y}/{m if m>9 else "0"+str(m)}/{d if d>9 else "0"+str(d)}"
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
    
    # Сохраняем даты
    df[['date']].to_csv('X.csv', index=False)
    # Сохраняем значения
    df[['value']].to_csv('Y.csv', index=False)
    
    print("Файл разделен на X.csv и Y.csv")

if __name__ == '__main__':

    # Лабораторная 1
    #df = scrape_val()
    #save_to_csv(df, )
    
    # Разделение на X и Y
    split_to_x_y('dataset.csv')
    
    # Разделение по годам
    #split_csv_by_years('dataset.csv', 'yearly_data')
    
    # Разделение по неделям
    #split_csv_by_weeks('dataset.csv', 'weekly_data')
    
    print("Все операции завершены!")