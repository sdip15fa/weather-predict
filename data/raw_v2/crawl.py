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
        for year in range(2005, 2023):
            for i in [["01", "07"], ["08", "15"], ["16", "23"], ["24", "30"]]:
                start_date = f"{year}06{i[0]}"
                end_date = f"{year}06{i[1]}"
        
                future = executor.submit(get_file_versions, start_date, end_date)
                futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            timestamps = future.result()
            download_files(timestamps, download_dir)

if __name__ == "__main__":
    main()
