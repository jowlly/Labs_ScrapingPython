import pandas as pd
import requests
from datetime import datetime

url = 'https://www.cbr-xml-daily.ru/archive/'
val ='CAD'
foryear = 2025

fromday = 1
frommonth = 1
years = 1
result=[]
for y in range(years):
    for m in range(frommonth,13):
        for d in range(fromday,32):
            cur_date = f"{foryear+y}/{m if m>9 else "0"+str(m)}/{d if d>9 else "0"+str(d)}"
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

filename = 'dataset.csv'
df.to_csv(filename,index=False)
