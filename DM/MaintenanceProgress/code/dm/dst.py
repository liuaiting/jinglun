# -*- coding:utf-8 -*-
import copy
import sys
import os
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)
import requests
import json
# from DB.database import select
"""

config_data = {
    'slotsNumber':{'slot_to_num':{'tel':0,'number':1},'num_to_slot':{0:'tel',1:'number'}},
    'inter_config':{'dm_res':{'answer':'','state':'','slots':{'slot_val':'','slot_name':''},'type':'','answer_id':''},
                    'inter_data':{'act_type':'','slot':{'slot_val':'','slot_name':''},'answer':'','state':[0,0],'action':'','query':{'tel':'','number':''},'answer_id':0}}
}
"""
class DST():
    
    def __init__(self,config,logger,user_info_slots):
        """
        config:
        slot_number:{'slot_to_num':{},'num_to_slot':{}}
        init_state = [0,0]

        """

        self.slot_number = config['slot_number']

        self.state = config['init_state']
        self.logger = logger

        self.query = copy.deepcopy(config['slot_info'])
        self.slot_info = config['slot_info']
        self.rejected_slots = []
        self.db_results = []

        for key in user_info_slots:

            slot_num = self.slot_number['slot_to_num'][key]
            self.state[slot_num] = 2
            self.slot_info[key] = user_info_slots[key]
            self.query[key] = user_info_slots[key]
            # print("query:",self.query)

            self.db_results = self.return_query_results(self.query)
            # print("init_db_result:",self.db_results)

            if len(self.db_results) == 0:
                self.query[key] = ""
                self.state[slot_num] = 1

    def return_query_results(self, query):

        # for key in query:
        #     if key == 'NUM' and query[key] != '':
        #         query[key] = int(query[key])

        post_data = json.dumps(query)
        url = "http://61.183.225.85:8083/CustomerService/queryMachineState"
        res_data = requests.post(url, data=post_data).text
        res_data = json.loads(res_data)

        try:
            return res_data['results']
        except:
            return []

    def update_state(self,nlu_result,dm_history):

        dm_res = dm_history['history_res']

        if dm_res != []:

            self.db_results = []
            self.rejected_slots = dm_history['rejected_slots']
            self.slot_info = dm_history['slots_info']
            self.query = dm_history['query']

            self.state = dm_res[-1]['state']
            old_acts = dm_res[-1]['action']

        self.logger.info('nlu_result is {}'.format(nlu_result))
        self.logger.info('slot info is {}'.format(self.slot_info))

        intent = nlu_result['intent']
        nlu_slots = nlu_result['slots']

        if intent == 'CONFIRM':

            try:
                act_type, act_slot = old_acts.split('_')
            except:
                self.logger.info("no confirm slots")
                exit()
            else:
                if act_type == 'ASK':
                    self.logger.info("dm ask a slot,but user confirm ?")
                    exit()
                slot_number = self.slot_number['slot_to_num'][act_slot]
                self.logger.info("confirm slot is {}".format(slot_number))
                self.state[slot_number] = 2
                self.query[act_slot] = self.slot_info[act_slot]

        elif intent == 'REJECT':

            try:
                act_type,act_slot = old_acts.split('_')
            except:
                self.logger.info('no reject slot_name or slot_value!')
                exit()
            else:
                if act_type == 'ASK':
                    # reject slot_name
                    self.rejected_slots.append('ASK_'+act_slot)
                else:
                    # reject slot_value
                    slot_number = self.slot_number['slot_to_num'][act_slot]
                    self.logger.info('reject slot is {}'.format(slot_number))
                    self.state[slot_number] = 0
                    self.slot_info[act_slot] = ""

        elif intent == 'INFORM':

            for i in range(0,len(nlu_slots)):

                slot = nlu_slots[i]
                slot_name,slot_val= slot['slot_name'],slot['slot_val']

                try:
                    slot_number = self.slot_number['slot_to_num'][slot_name]
                except:
                    self.logger.info("no inform slots")
                    exit()
                else:
                    self.logger.info("inform slot is {}".format(slot_number))
                    self.state[slot_number] = 2
                    self.slot_info[slot_name] = slot_val
                    self.query[slot_name] = slot_val
                    # print("query:", self.query)

                    self.db_results = self.return_query_results(self.query)
                    # print("db_results:",self.db_results)

                    if len(self.db_results) == 0:
                        self.query[slot_name] = ""
                        self.state[slot_number] = 1

        elif intent == 'OTHER':

            pass

        else:
            self.logger.info("intent error,intent:{} is not exist".format(intent))
            exit()

        self.logger.info('state is {}'.format(self.state))
        self.logger.info('reject slots is {}'.format(self.rejected_slots))

        return self.state,self.rejected_slots,self.slot_info,self.query,self.db_results

    def process(self,nlu_result,dm_history):

        state,rejected_slots,slot_info,query,db_results = self.update_state(nlu_result,dm_history)

        return state,rejected_slots,slot_info,query,db_results
