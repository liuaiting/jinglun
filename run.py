# -*- coding=utf-8 -*-
"""
Author: Liu Aiting
Date: 2018-10-12
"""
import sys
import os
sys.path.append(os.getcwd())
from dialogue_system import DialogueSystem

from flask import Flask, request
app = Flask(__name__)

import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger("h2")


@app.route("/", methods=['POST', 'GET'])
def run_main():
    """
    request_data: JSON string;
    {
         "user_info": {"user_id": "1", "info": {"TEL": "13211112222"}},
         "user_input": input("user:\t")
    }
    """

    request_data = request.json
    dialogue = DialogueSystem(request_data)
    dialogue.init_dialogue()
    dialogue.turn()
    return dialogue.sys_response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
