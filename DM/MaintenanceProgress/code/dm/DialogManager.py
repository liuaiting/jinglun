# -*- coding:utf-8 -*-
import json
import logging
from DM.MaintenanceProgress.code.dm.dst import DST
from DM.MaintenanceProgress.code.dm.policy import Policy
import logging
class DialogManager():

    def __init__(self,config,user_info_slots):



        """
        dstConfig:
        slot_info:{'slot_to_num':{},'num_to_slot':{}}
        init_state = [0,0]
        PolicyConfig:
        slot_info:{'slot_to_num':{},'num_to_slot':{}}
        init_state = [0,0]
        
        """

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%y-%m-%d %H:%M:%S',
            filename='DM/MaintenanceProgress/log/log.txt',
            filemode='w'
        )
        self.logger = logging.getLogger(__name__)

        self._dst = DST(config['dstConfig'],self.logger,user_info_slots)
        self._policy = Policy(config['policyConfig'],self.logger)



    def process(self,threadData):

        nlu_result = threadData['session_info']['turn_info']['nlu_result']
        dm_history = threadData['session_info']['history_info']['dm_history']

        state, rejected_slots, slot_info, query, db_results = self._dst.process(nlu_result,dm_history)
        state_flag = 0
        for i in range(len(state)):

            if state[i] == 1:
                state_flag = 1
                break

        threadData['session_info']['history_info']['dm_history']['rejected_slots'] = rejected_slots
        threadData['session_info']['history_info']['dm_history']['slots_info'] = slot_info
        threadData['session_info']['history_info']['dm_history']['query'] = query

        ans_num = len(db_results)

        if ans_num > 0:

            """
            ans_flag = 0

            for i in range(0,ans_num):

                if float(db_results[i][-1]) < 1:

                    ans_flag = 1


            if ans_flag == 0 and state_flag==0:

                threadData['session_info']['turn_info']['dm_result']['state'] = state
                threadData['session_info']['turn_info']['dm_result']['action'] = "NONE"
                threadData['session_info']['turn_info']['dm_result']['slots'] = {"slot_name":"","slot_val":""}
                threadData['session_info']['turn_info']['dm_result']['answers'] = []

                threadData['session_info']['history_info']['dm_history']['history_res'].append(threadData['session_info']['turn_info']['dm_result'])

                return threadData
            
            """

            if ans_num == 1 and state_flag == 0 :
                # print("db_result:",db_results)

                threadData['session_info']['turn_info']['dm_result']['state'] = state
                threadData['session_info']['turn_info']['dm_result']['action'] = "ANSWER"
                threadData['session_info']['turn_info']['dm_result']['slots'] = {"slot_name": "", "slot_val": ""}
                threadData['session_info']['turn_info']['dm_result']['answers'] = db_results
                threadData['session_info']['history_info']['dm_history']['history_res'].append(threadData['session_info']['turn_info']['dm_result'])

                return threadData

            elif (ans_num < 3 and state_flag == 0):

                threadData['session_info']['turn_info']['dm_result']['state'] = state
                threadData['session_info']['turn_info']['dm_result']['action'] = "SELECT"
                threadData['session_info']['turn_info']['dm_result']['slots'] = {"slot_name": "", "slot_val": ""}
                threadData['session_info']['turn_info']['dm_result']['answers'] = db_results
                threadData['session_info']['history_info']['dm_history']['history_res'].append(threadData['session_info']['turn_info']['dm_result'])

                return threadData


        dm_result = self._policy.process(state,query,slot_info,rejected_slots)

        threadData['session_info']['turn_info']['dm_result'] = dm_result
        threadData['session_info']['history_info']['dm_history']['history_res'].append(threadData['session_info']['turn_info']['dm_result'])

        return threadData