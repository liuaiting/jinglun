# 客服机器人

## 运行环境

python3.6

## 依赖库

tensorflow 1.4

jieba

sqlite3

## 运行方式

直接python运行即可

```bash
python run.py
```

##  系统模块

| 模块    | 模块目录 | 模块接口程序 | 模块负责人 |
| ------- | -------- | ------------ | ---------- |
| 系统    | 主目录   | run.py       |     |
| NLU模块 | NLU      | nlu.py       | 刘艾婷     |
| DM模块  | DM       | dm.py        | 李可       |
| NLG模块 | NLG      | nlg.py       | 陈嘉裕     |
| log     | log      | 日志文件     |            |

## 模块接口格式定义

接口格式均为标准json字符串！！！

接收输入：

```json
{
    "user_info": {"user_id": "1", "info":{"tel": ""}}, 
    "user_input": "user input string"
}
```

系统应答：

```json
{
    "sys_output": "system response string",
    "user_info": {"user_id": "1", "info":{"tel": ""}}, 
}
```

领域识别模块结果：

```json
{
  "dr_result":
  {
    "domain": "查询维修进度"
  }
}
```

自然语言理解模块结果：

```json
{
  "nlu_result":
  {
    "intent": "INFORM",
    "slots": [{"slot_name": "TEL", "slot_val": "13211112222"}]
  }
}
```

对话管理模块结果：

```json
{
  "dm_result":
  {
    "state": [0,0,0,0],
    "action": "CONFIRM_NUM",
    "slots": {"slot_name": "TEL","slot_val": "13211112222"},
    "answers": []
  }
}
```

自然语言生成模块结果：

```json
{
  "nlg_result":
  {
    "response": "请确认您的联系电话是13211112222吗",
    "message": "success"
  }
}
```