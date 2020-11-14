import logging
import os

# # create logger
# logger_name = "example"
# logger = logging.getLogger(logger_name)
# logger.setLevel(logging.DEBUG)
#
# # create file handler
# log_path = "./log.log"
# fh = logging.FileHandler(log_path)
# fh.setLevel(logging.WARN)
#
# # create formatter
# fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
# datefmt = "%a %d %b %Y %H:%M:%S"
# formatter = logging.Formatter(fmt, datefmt)
#
# # add handler and formatter to logger
# fh.setFormatter(formatter)
# logger.addHandler(fh)
#
# # print log info
# logger.debug('debug message')
# logger.info('info message')
# logger.warn('warn message')
#
# logger.error('error message')
# logger.critical('critical message')
# log_path = "log/1/2/dialogue.log"
# # log_dir = os.path.dirname(log_path)
# # if not os.path.exists(log_dir):
# #     os.makedirs(log_dir)
# #
# # logging.basicConfig(
# #     filename="log/{}/dialogue.log".format("1"),
# #     filemode="w",
# #     level=logging.DEBUG
# # )
# # logging.info("test")
# import logging
# import logging.config
# logging.config.fileConfig('logging.ini')
# # logger1 = logging.getLogger('h1')
# # logger1.debug('debug message')
# # logger1.info('info message')
# # logger1.warning('warning message')
# # logger1.error('error message')
# # logger1.critical('critical message')
#
# logger2 = logging.getLogger('h2')
# logger2.debug('debug message')
# logger2.info('info message')
# logger2.warning('warning message')
# logger2.error('error message')
# logger2.critical('critical message')
#
# logger3 = logging.getLogger('h3')
# logger3.debug('debug message')
# logger3.info('info message')
# logger3.warning('warning message')
# logger3.error('error message')
# logger3.critical('critical message')

import requests
import json


postdata = json.dumps({"NUM":"1811116","TEL":"18617142835","CODE":"","FMS":"287881407351"})
url = "http://61.183.225.85:8083/CustomerService/queryMachineState"

r = requests.post(url, data=postdata)

print(r.text)




