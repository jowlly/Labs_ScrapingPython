import os
import pandas as pd
import requests


url = 'https://www.cbr-xml-daily.ru/archive/'


id = 0
days = 32
months = 2
years = 1
result=[]
for y in range(years):
    for m in range(months):
        for d in range(days):
            cur_date = f"20{str(25-y)}/{m if m>9 else "0"+str(m+1)}/{d if d>9 else "0"+str(d+1)}"
            cur_url = url +cur_date +"/daily_json.js"

            print("Скрапинг страницы: ", cur_url)

            json_response = requests.get(cur_url).json()
            if("error" in json_response.keys()):
                print("Не найдено")

                continue
            print(json_response["Valute"]["USD"])
            result.append([str(json_response["Date"]).split("T")[0],json_response["Valute"]["USD"]["Value"]])

df = pd.DataFrame(result, columns = ["date", "value"])

filename = 'dataset.csv'
df.to_csv(filename,index=False)