# -*- coding:utf-8 -*-
import sys
import os
currentUrl = os.path.dirname(__file__)
parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))
sys.path.append(parentUrl)
import random
# from DB.database import select
import requests
import json

class Policy():

    def __init__(self,config,logger):
        """
        config:
        act_info:{'act_to_num':{'ask_tel':0,'ask_number':1,'confirm_tel':2,'confirm_number':3,'query':4,'answer':5,'none':6,'select':7},
                  'num_to_act':{0:'ask_tel',1:'ask_number,2:'confirm_tel',3:'confirm_number',4:'query',5:'answer',6:'none',7:'select'}}
        slot_info: {'slot_to_num':{},'num_to_slot':{}}
        policy_file:

        :param config:
        """

        self.logger = logger
        self.act_info = config['act_info']
        self.slot_num = config['slot_number']
        self.old_asked_slots = []
        self.policy_file = config['policy_file']
        self.policy_dict = {}

        self.make_policy_file()

        policy_flag = self.load_policy()
        if policy_flag: self.logger.info("load policy error!!")

    def list_to_num(self,state):

        i = len(state) -1
        num = 0
        while i >= 0:

            num = num*3 + state[i]
            i = i - 1

        return num

    def num_to_list(self,num):

        state = [0,0,0,0]
        i = 0
        while num > 0:
            state[i] = num % 3
            num = int(num / 3)
            i = i + 1

        return state

    # def return_query_results(self,query):
    #
    #     res = select(query)
    #     num = len(res)
    #     if num > 0:
    #         return res
    #     else:
    #         return []

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

    def make_policy_file(self):

        f = open(self.policy_file,'w')
        for i in range(0,81):

            state = self.num_to_list(int(i))
            query_flag = True
            actions = []
            for j in range(len(state)):
                if state[j] == 0:
                    slot = self.slot_num['num_to_slot'][str(j)]
                    act =  'ASK' + '_' + slot
                    actions.append(act)
                if state[j] == 1:
                    slot = self.slot_num['num_to_slot'][str(j)]
                    act = 'CONFIRM' + '_' + slot
                    actions.append(act)
                    query_flag = False

            if query_flag and state != [0,0,0,0]:
                actions.append('QUERY')

            tmp = str(state) + "\t" + (",").join(actions) + "\n"
            f.writelines(tmp)

        f.close()

    def load_policy(self):

        try:
            f = open(self.policy_file)
        except:
            self.logger.info("no policy file!")
            exit()
        else:
            for i,line in enumerate(f.readlines()):
                acts = line.strip().split('\t')[1]
                self.policy_dict[i] = acts
            f.close()

        return 0

    def get_action(self,state,rejected_slots):
        """
        :param state: 0,1,2,3,4...
        :return:
        """
        try:
            acts = self.policy_dict[state]
        except:
            self.logger.info("no policy dict!")
            exit()
        else:
            acts_list = acts.split(',')
            real_acts = list(set(acts_list)-set(rejected_slots))
            return real_acts

    def process(self,state,query,slot_info,rejected_slots):

        self.old_asked_slots = rejected_slots

        state_num = self.list_to_num(state)

        actions = self.get_action(state_num,rejected_slots)
        self.logger.info("dm action is {}".format(actions))

        dm_result = {
            'state':state,
            'action':'',
            'slots':{'slot_name':'','slot_val':''},
            'answers':[],
        }

        if actions == '':
            dm_result['action'] = 'NONE'
            return dm_result
        else:
            if 'QUERY' in actions:

                answers= self.return_query_results(query)
                query_num = len(answers)

                if query_num == 0:
                    dm_result['action'] = 'NONE'
                    return dm_result
                elif query_num == 1:
                    dm_result['action'] = 'ANSWER'
                    dm_result['answers'] = answers
                    return dm_result
                elif (query_num > 1 and query_num < 3):
                    dm_result['action'] = 'SELECT'
                    dm_result['answers'] = answers
                    return dm_result
                else:
                    actions.remove('QUERY')
                    if actions == []:
                        dm_result['action'] = 'SELECT'
                        dm_result['answers'] = answers
                        return dm_result

            action = random.choice(actions)
            dm_result['action'] = action

            '''
            if 'ASK' in action:
                self.old_asked_slots.append(action.split('_')[1])
            
            '''

            dm_result['slots']['slot_name'] = action.split('_')[1]
            dm_result['slots']['slot_val'] = slot_info[dm_result['slots']['slot_name']] if 'CONFIRM' in action else ''

            return dm_result
