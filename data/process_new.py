import csv
import datetime
import pandas as pd
import numpy as np

input_file = 'TaiMoShan.csv'
output_file = 'processed_data.csv'

# Function to separate date time into year, month, date, time, and minute
def separate_datetime(date_time):
    year = date_time[:4]
    month = date_time[4:6]
    date = date_time[6:8]
    return year, month, date

# Read the CSV file and process the data
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Read the header row
    data = []
    for row in reader:
        date_time = row[0]
        year, month, date = separate_datetime(date_time)
        row[1] = str(int(float(row[1]))).zfill(4)
        time = row[1][:2]
        minute = row[1][2:4]
        temperature = float(row[2]) / 10
        humidity = row[3]
        wind_speed = row[4]
        wind_direction = row[5]
        rainfall = row[6]
        data.append([date_time, year, month, date, time, minute, temperature])
    df = pd.DataFrame(data, columns=['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature'])
    df.dropna(inplace=True)
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors="coerce")
    average_temperatures = df.groupby(['Year', 'Month', 'Date'])['Temperature'].mean().to_dict()
    data2 = []
    for row in data:
        try:
            seven_days_average = [average_temperatures[tuple((datetime.datetime(int(row[1]), int(row[2]), int(row[3])) - datetime.timedelta(days=i)).strftime('%Y-%m-%d').split("-"))] for i in range(0, 8)]
            row.append(seven_days_average[0])
            row.append(seven_days_average[1])
            row.append(seven_days_average[2])
            row.append(np.mean(seven_days_average))
            data2.append(row)
        except KeyError:
            pass
    df = pd.DataFrame(data2, columns=['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature', 'Previous Day Average', 'Two Days Before Average', 'Three Days Before Average', 'Last 7 Days Average'])
    df.to_csv(output_file, index=False)
print("Data processing complete.")
