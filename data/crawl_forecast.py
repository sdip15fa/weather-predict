import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
import os

CSV_FILE = "forecast.csv"
START_DATE = "2008-01-01 00:01:00"
URL_TEMPLATE = "https://maps.weather.gov.hk/ocf/dat/SKG.xml?v={date}"
LOCK = threading.Lock()

df = None

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE, parse_dates=["DateTime"])
else:
    df = pd.DataFrame(columns=["DateTime", "ForecastTemperature",
                               "ForecastRelativeHumidity", "ForecastWindDirection", "ForecastWindSpeed"])


def fetch_data(date: datetime):
    global df
    unix_time = int(time.mktime(date.timetuple()))
    response = requests.get(URL_TEMPLATE.format(date=unix_time))
    data = response.json()

    if "HourlyWeatherForecast" in data:
        hourly_forecasts = []
        for hour_data in data["HourlyWeatherForecast"]:
            timestamp = datetime.strptime(
                hour_data["ForecastHour"] + "01", "%Y%m%d%H%M")
            if not (timestamp in pd.to_datetime(df["DateTime"]).unique()):
                try:
                    hourly_forecasts.append({
                        "DateTime": timestamp,
                        "ForecastTemperature": getattr(hour_data, "ForecastTemperature", None),
                        "ForecastRelativeHumidity": getattr(hour_data, "ForecastRelativeHumidity", None),
                        "ForecastWindDirection": getattr(hour_data, "ForecastWindDirection", None),
                        "ForecastWindSpeed": getattr(hour_data, "ForecastWindSpeed", None)
                    })
                except KeyError as e:
                    raise e
            df = pd.concat([df, pd.DataFrame(hourly_forecasts)])
            # print(df, hourly_forecasts)


def main():
    global df
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.now()
    date = start_date

    threads = []
    while date <= end_date:
        thread = threading.Thread(target=fetch_data, args=(date,))
        thread.start()
        threads.append(thread)

        date += timedelta(days=7)

        if len(threads) >= 5:
            print(len(df))
            threads[0].join()
            threads.pop(0)

    for thread in threads:
        thread.join()

    print("to csv")
    df = df.sort_values(by="DateTime").drop_duplicates(subset=["DateTime"])
    df.to_csv(CSV_FILE, index=False)


if __name__ == "__main__":
    main()
