import json

config= {'dstConfig':{'slot_number':{'slot_to_num':{'TEL':1,'NUM':0,'CODE':2,'FMS':3},'num_to_slot':{1:'TEL',0:'NUM',2:'CODE',3:'FMS'}},'init_state':[0,0,0,0],'slot_info':{"TEL":'','NUM':'','CODE':'','FMS':''}},
         'policyConfig': {'act_info': {
                                      'act_to_num': {'ASK_TEL': 1, 'ASK_NUM': 0, 'CONFIRM_TEL': 3, 'CONFIRM_NUM': 2, 'ASK_CODE':4, 'CONFIRM_CODE':5, 'ASK_FMS':6, 'CONFIRM_FMS':7,
                                                     'QUERY': 8,'ANSWER': 9, 'NONE': 10, 'SELECT': 11},
                                      'num_to_act': {0:'ASK_NUM',1:'ASK_TEL',2:'CONFIRM_NUM',3:'CONFIRM_TEL',4:'ASK_CODE',5:'CONFIRM_CODE',6:'ASK_FMS',7:'CONFIRM_FMS',
                                                     8:'QUERY',9:'ANSWER',10:'NONE',11:'SELECT'}
                                      },
                          'slot_number':{'slot_to_num':{'TEL':1,'NUM':0,'CODE':2,'FMS':3},'num_to_slot':{1:'TEL',0:'NUM',2:'CODE',3:'FMS'}},
                          'policy_file':'DM/MaintenanceProgress/data/parameters.txt'
                         }
         }

a = json.dumps(config)
f = open('../../data/config.txt','w')
f.write(a)
f.close()

print ("yes")