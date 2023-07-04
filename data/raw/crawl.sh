#!/bin/bash
for year in 2023; do
    mkdir $year;
    for month in 07; do
        mkdir "$year/$month";
        for date in 04; do
            mkdir "$year/$month/$date";
            for hour in 18; do
                mkdir "$year/$month/$date/$hour";
                for time in {00..59}; do
                    filename="$year/$month/$date/$hour/$time.csv"
                    if [ -f "$filename" ]; then
                        continue;
                    fi;
                    wget -nc "https://s3-ap-southeast-1.amazonaws.com/historical-resource-archive/${year}/${month}/${date}/https%253A%252F%252Fdata.weather.gov.hk%252FweatherAPI%252Fhko_data%252Fregional-weather%252Flatest_1min_temperature.csv/${hour}${time}" -O "$year/$month/$date/$hour/$time.csv" &
                    sleep 0.1;
                done;
            done;
        done;
    done;
done;
