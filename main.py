import pandas as pd
import requests

url = 'https://www.cbr-xml-daily.ru/archive/'
val ='CAD'
days = 32
months = 13
years = 2
result=[]
for y in range(years):
    for m in range(1,months):
        for d in range(1,days):
            cur_date = f"{2024+y}/{m if m>9 else "0"+str(m)}/{d if d>9 else "0"+str(d)}"
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