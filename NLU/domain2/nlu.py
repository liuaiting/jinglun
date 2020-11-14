# -*- coding: utf-8 -*-
"""
NLU interface.

Date: 2018-09-29
Author: Liu Aiting
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re
import os
import logging

import tensorflow as tf

from NLU.domain2 import hlstm
from NLU.domain2.utils import model_helper
from NLU.domain2.utils import misc_utils as utils


root_path = "NLU/domain2"
model_path = os.path.join(root_path, "model")
hparams_path = os.path.join(model_path, "hparams")


def tokenize(string):
    """
    Tokenize the user input.

    Example:
        string: "sd123好的123啊是地方sd  123 what  hhh"
        output: "sd123 好 的 123 啊 是 地 方 sd 123 what hhh"

    Args:
        string: str, user input.
    Return:
        output: str, char level tokenize result, space separated.
    """
    assert type(string) == str
    output = re.sub(r'([\u4e00-\u9fa5])', r' \1 ', string)
    output = re.sub(r'([a-zA-Z0-9]+)', r' \1 ', output).strip()
    output = re.sub(r'\s+', r' ', output).strip()
    return output


def get_entity(seq, tag):
    """Get slot values given original sequence and sequence labeling result."""
    tag_split = [a.split("-") for a in tag]
    index = 0
    entity = []
    while index < len(seq):
        if tag_split[index][0] == "B":
            start = index
            index += 1
            while index < len(seq) and tag_split[index][0] == "I" and tag_split[index][1] == tag_split[start][1]:
                index += 1
            if index < len(seq):
                end = index
            else:
                end = index + 1

            slot_val = str("".join(seq[start:end]))
            slot_name = str(tag_split[start][1])
            entity_ = {
                "slot_val": slot_val,
                "slot_name": slot_name
            }
            entity.append(entity_)
        else:
            index += 1

    return entity


def start_sess_and_load_model(infer_model, ckpt_path):
    """Start session and load model."""
    sess = tf.Session(
        graph=infer_model.graph, config=utils.get_config_proto())
    with infer_model.graph.as_default():
        loaded_infer_model = model_helper.load_model(
            infer_model.model, ckpt_path, sess, "infer")
    return sess, loaded_infer_model


def load_nlu_model():
    """Load nlu model."""
    # load hparams
    ckpt = tf.train.latest_checkpoint(model_path)
    hparams_values = json.load(open(hparams_path))
    hparams = tf.contrib.training.HParams(**hparams_values)

    # Create model
    model_creator = hlstm.HLSTM
    infer_model = model_helper.create_infer_model(model_creator, hparams, mode="infer")

    # TensorFlow model
    sess, loaded_infer_model = start_sess_and_load_model(infer_model, ckpt)

    return sess, loaded_infer_model, infer_model


class NLU(object):
    def __init__(self, stream_data):
        try:
            self.stream_data = json.loads(stream_data)
        except TypeError:
            raise TypeError("user_request must be a standard JSON string!")

        self.logger = self.init_logger()
        self.user_input = self.stream_data["request_data"]["user_input"]
        self.nlu_result = {}
        self.logger.info("new an instance of NLU class success.")

    def init_logger(self):
        log_path = os.path.join(root_path, "log/nlu.log")
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%y-%m-%d %H:%M:%S',
            filename=log_path,
            filemode='w'
        )
        logger = logging.getLogger("NLULogger")
        return logger

    def nlu_interface(self, sess, loaded_infer_model, infer_model):
        # Infer data
        infer_data = [tokenize(self.user_input)]
        sess.run(infer_model.iterator.initializer,
                 feed_dict={infer_model.data_placeholder: infer_data})
        while True:
            try:
                seq, seq_len, pred_label, pred_intent = loaded_infer_model.infer(sess)
                seq_len = seq_len.tolist()[0]
                seq = infer_data[0]
                pred_label = b" ".join(pred_label.tolist()[0][:seq_len]).decode("utf-8")
                pred_intent = pred_intent.tolist()[0].decode("utf-8")
                slots = get_entity(seq.split(), pred_label.split())

                self.nlu_result = {
                    "intent": pred_intent,
                    "slots": slots,
                }

                self.__check_nlu_result()
                self.__update_stream_data()
                self.__update_nlu_history()

            except tf.errors.OutOfRangeError:
                break

    def __check_nlu_result(self):
        intent = self.nlu_result["intent"]
        if intent == "INFORM" and not self.nlu_result["slots"]:
            raise ValueError("INFORM intent but empty slots")
        if intent == "CONFIRM" or intent == "REJECT":
            self.nlu_result["slots"] = []

    def __update_stream_data(self):
        try:
            self.stream_data["session_info"]["turn_info"]["nlu_result"] = self.nlu_result
            self.logger.info("update nlu result -> {}.".format(self.nlu_result))

        except ValueError:
            raise ValueError("update nlu result error.")

    def __update_nlu_history(self):
        try:
            for slot_dict in self.nlu_result["slots"]:
                slot_name = slot_dict["slot_name"]
                slot_val = slot_dict["slot_val"]
                self.stream_data["session_info"]["history_info"]["nlu_history"][slot_name] = slot_val

                self.logger.info("update nlu history {}: {}.".format(slot_name, slot_val))

        except ValueError:
            raise ValueError("update nlu history error.")


#
# if __name__ == "__main__":
#     data = {
#         "user_id": "1",
#
#         "request_data":
#             {
#                 "user_info": {"user_id": "1", "info": {"TEL": "1323323232"}},
#                 "user_input": "快递单号是343434343"
#             },
#
#         "session_info":
#             {
#                 "session_id": "0",
#                 "dr_result": {},
#                 "turn_info":
#                     {
#                         "turn_id": "1",
#                         "turn_input": "快递单号是343434343",
#                         "nlu_result": {},
#                         "dm_result": {},
#                         "nlg_result": {},
#                         "response_data": {}
#                     },
#
#                 "history_info":
#                     {
#                         "dm_history": {},
#                         "nlu_history": {},
#                         "nlg_history": {}
#                     }
#             }
#     }
#
#     sess, loaded_infer_model, infer_model = load_nlu_model()
#
#     n = NLU(data)
#     n.nlu_interface(sess, loaded_infer_model, infer_model)
