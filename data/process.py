import csv

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
        data.append([year, month, date, time, minute, air_temperature])

# Write the processed data to a new CSV file
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Year', 'Month', 'Date', 'Time', 'Minute', 'Air Temperature'])
    writer.writerows(data)

print("Data processing complete.")
