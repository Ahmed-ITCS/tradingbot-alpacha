import requests

url = "https://paper-api.alpaca.markets/v2/positions"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": "PKRGS52QTJ0TEN1DPPS5",
    "APCA-API-SECRET-KEY": "afqTWIfxp3bcRGHXtd2kfSzqyIbcBy07v438xmVK"
}

response = requests.delete(url, headers=headers)

print(response.text)