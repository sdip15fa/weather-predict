import csv
import os
import glob

# Create an empty list to store the rows
rows = []

# Iterate over all the files in the directory
for filename in glob.glob("./raw/**/*.csv", recursive=True):
    if filename.endswith('.csv'):  # Only process CSV files
        file_path = filename
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            # Check if the CSV file is empty or doesn't contain the required column
            if csv_reader.fieldnames is None or 'Automatic Weather Station' not in csv_reader.fieldnames:
                continue
            
            # Iterate over each row in the CSV file
            for row in csv_reader:
                if row['Automatic Weather Station'] == 'Tai Mo Shan':
                    rows.append(row)

# Define the output file path
output_file = 'data.csv'

# Write the combined rows to the output file
with open(output_file, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

print("Rows extracted and combined successfully!")
