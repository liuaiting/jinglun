# -*- coding: utf-8 -*-
"""
Dialogue System Model.

Author: Liu Aiting
Date:   2018-10-12
E-mail: liuaiting@bupt.edu.cn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from NLU.domain1 import nlu as nlu_domain1
from DM.MaintenanceProgress.code.dm.DialogManager import DialogManager
from NLG.nlg import NLG

import logging
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger('h3')

COOKIE_PATH = "COOKIE"
COOKIE_NAME = "cookie.json"


def dr(stream_data):
    """

    :param stream_data:
    :return: default 查询维修进度
    """
    try:
        stream_data = json.loads(stream_data)
    except TypeError:
        raise TypeError("stream data must be a standard JSON string!")

    dr_result = {"domain": "查询维修进度"}

    if stream_data["session_info"]["dr_result"] == {}:
        stream_data["session_info"]["dr_result"] = dr_result
    else:
        raise ValueError("Perform domain recognition only in the first turn.")
    return json.dumps(stream_data)


class DialogueSystem(object):
    def __init__(self, user_request):
        """
        :param user_request: JSON string
                            {"user_info": {"user_id": "1", "info": {"TEL": "1323323232"}},
                            "user_input": "user input string"}
        """

        try:
            self.user_request = json.loads(user_request)
        except TypeError:
            raise TypeError("user_request must be a standard JSON string!")

        self.user_id = str(self.user_request["user_info"]["user_id"])
        self.user_input = str(self.user_request["user_input"])
        self.user_info = str(self.user_request["user_info"])

        self.sys_response = None
        self.stream_data = None
        self.user_cookie_path = os.path.join(COOKIE_PATH, str(self.user_id))
        self.user_cookie_file = os.path.join(self.user_cookie_path, "cookie.json")
        if not os.path.exists(self.user_cookie_path):
            os.makedirs(self.user_cookie_path)

    def is_dialogue_over(self):
        stream_data = json.loads(self.stream_data)
        dm_action = stream_data["session_info"]["turn_info"]["dm_result"]["action"]
        if dm_action == "ANSWER" or dm_action == "None" or dm_action == "SELECT":
            return True
        else:
            return False

    def init_dialogue(self):
        if not os.path.isfile(self.user_cookie_file):
            self.init_stream_data()
            self.stream_data = dr(self.stream_data)
            logger.info("new user, init dialogue cookie.json")
        else:
            self.load_stream_data()
            if self.is_dialogue_over():
                self.init_stream_data()
                self.stream_data = dr(self.stream_data)
                logger.info("old user, last dialogue is over.")
            else:
                self.reset_turn_info()
                logger.info("old user, continue last dialogue, reset turn info.")

    def reset_turn_info(self):
        stream_data = self.stream_data
        stream_data["request_data"] = self.user_request
        stream_data["session_info"]["turn_info"]["turn_id"] = str(int(
            stream_data["session_info"]["turn_info"]["turn_id"]) + 1)
        stream_data["session_info"]["turn_info"]["turn_input"] = self.user_input
        stream_data["session_info"]["turn_info"]["nlu_result"] = {}
        stream_data["session_info"]["turn_info"]["dm_result"] = {}
        stream_data["session_info"]["turn_info"]["nlg_result"] = {}
        stream_data["session_info"]["turn_info"]["response_data"] = {}
        self.stream_data = stream_data

    def load_stream_data(self):
        f = open(self.user_cookie_file)
        stream_data = json.load(f)
        f.close()

        self.stream_data = json.dumps(stream_data)

    def init_stream_data(self):

        f = open("data_format/init_info_format.json")
        stream_data = json.load(f)
        f.close()
        stream_data["request_data"] = self.user_request
        stream_data["user_id"] = self.user_id
        stream_data["session_info"]["turn_info"]["turn_input"] = self.user_input
        stream_data["session_info"]["turn_info"]["turn_id"] = "1"

        self.stream_data = json.dumps(stream_data)

    def turn(self):
        domain = json.loads(self.stream_data)["session_info"]["dr_result"]["domain"]
        if domain == "查询维修进度":
            self.perform_nlu()
            self.perform_dm()
            self.perform_nlg()
            self.get_sys_response()
            f = open(self.user_cookie_file, "w")
            json.dump(json.loads(self.stream_data), f, indent=4, ensure_ascii=False)
            f.close()
        else:
            logger.error("dialogue turn - domain error!")

    def domain_recognition(self):
        """
        Perform domain recognition in first turn.
        """
        # TODO: add domain recognition module
        self.stream_data = dr(self.stream_data)

    def perform_nlu(self):
        """
        Select the appropriate module according to dr_result.
        :return:
        """
        # LOAD MODEL
        sess1, nlu_loaded_model1, nlu_model1 = nlu_domain1.load_nlu_model()

        stream_data = json.loads(self.stream_data)
        dr_result = stream_data["session_info"]["dr_result"]
        domain = dr_result["domain"]

        if domain == "查询维修进度":
            n = nlu_domain1.NLU(self.stream_data)
            self.stream_data = n.nlu_interface(sess1, nlu_loaded_model1, nlu_model1)
        else:
            raise ValueError("not other domain for now.")

    def perform_dm(self):
        # TODO: DM module
        config = json.load(open("DM/MaintenanceProgress/data/config.txt"))
        threadData = json.loads(self.stream_data)
        user_info_slots = threadData['request_data']['user_info']['info']
        dialogManager = DialogManager(config, user_info_slots)
        self.stream_data = dialogManager.process(threadData)

    def perform_nlg(self):
        # TODO: NLG module
        file_name = 'NLG/nlg_data.txt'
        nlg = NLG(file_name)
        self.stream_data = nlg.gen_sen(self.stream_data)

    def get_sys_response(self):
        stream_data = json.loads(self.stream_data)
        self.sys_response = json.dumps({
            "user_info": self.user_info,
            "sys_output": stream_data["session_info"]["turn_info"]["nlg_result"]["response"]})
