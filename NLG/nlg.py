 # -*- coding: UTF-8 -*-
import re
import random
import traceback
import json
def read_template(file_name):
	template={'ASK':[],'CONFIRM':[],'NONE':[],'ANSWER':[],'SELECT':[]}
	for line in open(file_name).readlines():
		line=line.strip().split(':')
		template[line[0]].append(line[1])
	# print(template)
	return template

class NLG(object):
	def __init__(self, file_name):
		self.template=read_template(file_name)
		self.slot=['TEL','NUM','CODE','FMS']
		self.slot2=['NONE','ANSWER','SELECT']
		self.name={'TEL':'手机号码','NUM':'维修单号','CODE':'整机条码','FMS':'快递单号'}
		self.action=['ASK','CONFIRM']
	def make_dm_result(self):	
		'''
		随机产生dm_result，做测试用
		'ASK_TEL','ASK_NUM','ASK_CODE','ASK_FMS'
			'CONFIRM_TEL','CONFIRM_NUM','CONFIRM_CODE','CONFIRM_FMS'
		'''	
		act=random.sample(self.action,1)[0]
		ss=random.sample(self.slot,1)[0]
		slots={'slot_val':'222','slot_name':ss}
		dm_result={'action':act+'_'+ss,'slots':slots}
		return dm_result
	def make_dm_result2(self):	
		'''
		随机产生dm_result，做测试用,['NONE','ANSWER','SELECT']
		'''	
		act=random.sample(self.slot2,1)[0]
		answers=[(12314,1245365427,'ZPA34612786','4721646ggf7236',0.45),(12314,1245365427,'ZPA34612786','4721646ggf7236',0.8),
		(12314,1245365427,'ZPA34612786','4721646ggf7236',0.4),(12314,1245365427,'ZPA34612786','4721646ggf7236',0.9),
		(12314,1245365427,'ZPA34612786','4721646ggf7236',0.55),(12314,1245365427,'ZPA34612786','4721646ggf7236',0.0)]
		answer_id=[0,1]
		dm_result={'action':act,'answers':answers}
		return dm_result
	def gen_sen(self,interface_json):
		'''
		根据模板产生对应的应答
		'''
		dm_result=interface_json["session_info"]["turn_info"]["dm_result"]


		if "history_action" in interface_json["session_info"]["history_info"]["nlg_history"]:
			interface_json["session_info"]["history_info"]["nlg_history"]["history_action"].append(dm_result['action'])
		else:
			interface_json["session_info"]["history_info"]["nlg_history"]["history_action"]=[dm_result['action']]
		

		act=dm_result['action'].split('_')[0]
		if len(dm_result['action'].split('_'))>1:
			slot=dm_result['action'].split('_')[1]
		else:
			slot=''
		# print(act,slot)
		if act=='ASK': 
			sen=random.sample(self.template['ASK'],1)[0]
			sen=sen.replace('y',self.name[slot])
			nlg_result={'response':sen,'message':'success'}
		elif act=='CONFIRM':
			sen=random.sample(self.template['CONFIRM'],1)[0]
			sen=sen.replace('y',self.name[slot]).replace('x',dm_result['slots']['slot_val'])
			nlg_result={'response':sen,'message':'success'}
		elif act=='NONE':
			sen=random.sample(self.template['NONE'],1)[0]
			nlg_result={'response':sen,'message':'success'}
		elif act=='ANSWER':
			sen=random.sample(self.template['ANSWER'],1)[0]
			if len(dm_result['answers'])==0:
				nlg_result={'response':'ERROR!!','message':'dm_result answers is empty'}
			else:
				an=dm_result['answers'][0]
					
				res = [an["NUM"],an["TEL"],an["CODE"],an["FMS"],an["ans"]]
				for i in range(5):
					sen=sen.replace('x'+str(i+1),str(res[i]))
				nlg_result={'response':sen,'message':'success'}
		elif act=='SELECT':
			sen_temp=random.sample(self.template['SELECT'],1)[0]
			if len(dm_result['answers'])==0:
				nlg_result={'response':'ERROR!!','message':'dm_result answers is empty'}
			else:
				if len(dm_result['answers'])>5:
					ans=dm_result['answers'][:5]
				else:
					ans=dm_result['answers']
				# print(ans)
				res = []
				for an in ans:
					temp=[an["NUM"],an["TEL"],an["CODE"],an["FMS"],an["ans"]]
					res.append(temp)
				sen='为您查询到如下结果:\n'
				for k in range(len(res)):
					an=res[k]
					temp=sen_temp
					# print(an)
					temp=temp.replace('k',str(k+1))
					for i in range(5):
						temp=temp.replace('x'+str(i+1),str(an[i]))
					temp=temp+'\n'
					sen+=temp
					
				nlg_result={'response':sen,'message':'success'}

		else:
			nlg_result={'response':'ERROR!!','message':'ERROR!! act is out of range!'}
			
		interface_json["session_info"]["turn_info"]["nlg_result"]=nlg_result
		# file = open('1.json','w')
		# json.dump(interface_json,file,indent=4,ensure_ascii=False)
		# print(nlg_result)
		return json.dumps(interface_json)



# if __name__ =="__main__":
# 	file_name='nlg_data.txt'
# 	nlg=NLG(file_name)
# 	# dm_result=nlg.make_dm_result()
# 	# dm_result=nlg.make_dm_result2()
# 	# print(dm_result)
# 	json_file='interface2.json'
# 	interface_json=json.load(open(json_file))
# 	# print(interface)
# 	interface_json=nlg.gen_sen(interface_json)
# 	# print(nlg_result)
# 	# print(interface_json)