 # -*- coding: UTF-8 -*-
import re
import random
import traceback
import json
from nlg import NLG


if __name__ =="__main__":
	file_name='nlg_data.txt'
	nlg=NLG(file_name)
	json_file='interface2.json'
	interface_json=json.load(open(json_file))
	interface_json=nlg.gen_sen(interface_json)
