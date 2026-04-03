import pandas as pd
import requests
from pathlib import Path
import sys

# add root to path so scripts folder is accessible
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from scripts.data_clean_utils import perform_data_cleaning

# path for data
data_path = root_path / "data" / "raw" / "swiggy.csv"

# prediction endpoint - local server
predict_url = "http://127.0.0.1:8000/predict"

# sample row for testing the endpoint
sample_row = pd.read_csv(data_path).dropna().sample(1)
print("The target value is", sample_row.iloc[:, -1].values.item().replace("(min) ", ""))

# remove the target column
raw_data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]])

# clean the raw data
cleaned_data = perform_data_cleaning(raw_data)

# convert to dict for API request
data = cleaned_data.squeeze().to_dict()
print("Cleaned data being sent:", data)

# get the response from API
response = requests.post(url=predict_url, json=data)

print("The status code for response is", response.status_code)

if response.status_code == 200:
    print(f"The prediction value by the API is {response.json()['predicted_delivery_time']:.2f} min")
else:
    print("Error:", response.text)