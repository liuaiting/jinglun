import requests
import os
import json
os.environ['NO_PROXY'] = '127.0.0.1'

while True:
    request = {
             "user_info": {"user_id": input(), "info": {}},
             "user_input": input("user:")
        }
    request_data = json.dumps(request)

    r = requests.post("http://127.0.0.1:5000", json=request_data)
    # print(r.text)
    print(r.text)