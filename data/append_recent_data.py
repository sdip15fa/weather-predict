import csv
import pandas as pd

input_file = 'processed_data.csv'
output_file = 'TaiMoShan.csv'

# Read the existing rows from the "TaiMoShan.csv" file
existing_datetime = []
with open(output_file, 'r') as existing_csv:
    reader = csv.reader(existing_csv)
    next(reader)  # Skip the header row
    for row in reader:
        date = row[0]
        time = row[1]
        # print(date, time)
        existing_datetime.append([row[0], row[1]])

# Append the new data to the "TaiMoShan.csv" file
with open(input_file, 'r') as input_csv, open(output_file, 'a', newline='') as output_csv:
    reader = csv.DictReader(input_csv)
    writer = csv.writer(output_csv)

    output_df = pd.read_csv(output_file)

    for row in reader:
        year = int(row['Year'])
        month = int(row['Month'])
        day = int(row['Date'])
        hour = int(row['Time'])
        minute = int(row['Minute'])

        if year >= 2023 and month >= 6 and day >= 26 and row['Temperature'] != "N/A":
            # Extract the required fields from the row
            temperature = float(row['Temperature']) * 10
            humidity = ''  # Add your logic to extract humidity from the row
            wind_speed = ''  # Add your logic to extract wind speed from the row
            wind_direction = ''  # Add your logic to extract wind direction from the row

            # Format the date, hour, and minute with leading zeros if needed
            date_str = f'{year}{month:02d}{day:02d}'
            time_str = f'{hour:02d}{minute:02d}'

            if not len(output_df[(output_df["YYYYMMDD"].astype(float) == float(date_str)) & (output_df["HHMM"].astype(float) == float(time_str))]):
                # Append the extracted data to the output file
                writer.writerow([
                    date_str,
                    time_str,
                    temperature,
                    humidity,
                    wind_speed,
                    wind_direction,
                    '',  # Assuming rainfall data is not available in the input file, set it as empty
                    wind_direction
                ])

print("Data appended successfully.")
