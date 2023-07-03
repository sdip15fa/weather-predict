import csv
import pandas as pd
import sys

input_file = sys.argv[1] if len(sys.argv) > 1 else 'TaiMoShan.csv'
output_file = sys.argv[2] if len(sys.argv) > 2 else 'processed_data.csv'

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
        wind_direction = row[5] or row[7]
        rainfall = row[6]
        data.append([date_time, year, month, date, time, minute, temperature, wind_speed, wind_direction, rainfall, humidity])
    df = pd.DataFrame(data, columns=['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature', 'Wind Speed', 'Wind Direction', 'Rainfall', 'Humidity'])
    df.dropna(inplace=True)
    df['DateTime'] = pd.to_datetime(df['DateTime'].astype(float).astype(int).astype(str) + df['Time'].astype(str) + df['Minute'].astype(str), format='%Y%m%d%H%M')
    df.to_csv(output_file, index=False)
print("Data processing complete.")
