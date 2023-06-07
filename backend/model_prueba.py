import requests

# Define the input data
input_data = {"data": [[1]], "max_token": 100}

# Send the request to the TorchServe endpoint
response = requests.post("http://localhost:8080/predictions/fyodor", json=input_data)

# Get the predicted output
if response.status_code == 200:
    output = response.text
    print(output)
else:
    print("Error:", response.text)
