槽对应情况为"TEL": 1, "NUM": 0, "CODE": 2, "FMS": 3，详情可见配置文件，配置文件在data/config.txt
DM整体模块封装在DialogManager中
在没有nlu时候可以cmd输入意图和槽信息，运行run.py即可，调用时候可参考run.py的使用方法

值得注意的点：
1 由于没有数据库，人为设定查询结果 random的 可能为 0、1、2 任何一种情况，系统状态出现2时候就会查数据
2 任何异常情况都会exit，运行过程中可见log。 位置在log/log.txt。