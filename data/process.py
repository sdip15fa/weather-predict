import csv
import datetime
import pandas as pd
import numpy as np

input_file = 'data.csv'
output_file = 'processed_data.csv'



# Function to separate date time into year, month, date, time, and minute
def separate_datetime(date_time):
    year = date_time[:4]
    month = date_time[4:6]
    date = date_time[6:8]
    time = date_time[8:10]
    minute = date_time[10:]
    return year, month, date, time, minute

# Read the CSV file and process the data
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Read the header row
    data = []
    for row in reader:
        date_time = row[0]
        weather_station = row[1]
        air_temperature = row[2]
        year, month, date, time, minute = separate_datetime(date_time)
        data.append([date_time, year, month, date, time, minute, air_temperature])
    df = pd.DataFrame(data, columns=['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature'])
    #df.dropna(inplace=True)
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors="coerce")
    average_temperatures = df.groupby(['Year', 'Month', 'Date'])['Temperature'].mean().to_dict()
    # print(average_temperatures)
    data2 = []
    for row in data:
        try:    
            seven_days_average = [average_temperatures[tuple((datetime.datetime(int(row[1]), int(row[2]), int(row[3])) - datetime.timedelta(days=i)).strftime('%Y-%m-%d').split("-"))] for i in range(1,8)]
            row.append(seven_days_average[0])
            row.append(seven_days_average[1])
            row.append(seven_days_average[2])
            row.append(np.mean(seven_days_average))
        except KeyError:
            pass
        finally:
            data2.append(row)
        
    df = pd.DataFrame(data2, columns=['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature', 'Previous Day Average', 'Two Days Before Average', 'Three Days Before Average', 'Last 7 Days Average'])
    df.to_csv(output_file)
print("Data processing complete.")
