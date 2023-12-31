import requests
import os
import concurrent.futures

def get_file_versions(start_date, end_date):
    url = f"https://api.data.gov.hk/v1/historical-archive/list-file-versions?url=https%3A%2F%2Fdata.weather.gov.hk%2FweatherAPI%2Fhko_data%2Fregional-weather%2Flatest_1min_temperature.csv&start={start_date}&end={end_date}"
    response = requests.get(url)
    data = response.json()
    return data["timestamps"]

def download_file(download_url, timestamp, download_dir):
    response = requests.get(download_url)
    with open(os.path.join(download_dir, f"{timestamp}_data.csv"), "wb") as f:
        f.write(response.content)

def download_files(timestamps, download_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for timestamp in timestamps:
            download_url = f"https://api.data.gov.hk/v1/historical-archive/get-file?url=https%3A%2F%2Fdata.weather.gov.hk%2FweatherAPI%2Fhko_data%2Fregional-weather%2Flatest_1min_temperature.csv&time={timestamp}"
            futures.append(executor.submit(download_file, download_url, timestamp, download_dir))
        concurrent.futures.wait(futures)

def main():
    download_dir = "downloads"
    os.makedirs(download_dir, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for year in [2023]: # range(2020, 2023):
            for month in ["06"]:  # ["01", "02", "03, '04", '05', '06', '07', '08', '09', '10', '11', '12']:        
                for i in [["26", "28"]]: # [["01", "07"], ["08", "15"], ["16", "23"], ["24", "30"]]:
                    start_date = f"{year}{month}{i[0]}"
                    end_date = f"{year}{month}{i[1]}"
                    future = executor.submit(get_file_versions, start_date, end_date)
                    futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                timestamps = future.result()
                download_files(timestamps, download_dir)
            except KeyError as e:
                print(e)
                pass

if __name__ == "__main__":
    main()
