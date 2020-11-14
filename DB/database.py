import sqlite3
import pandas as pd
from contextlib import closing


def create_db():

    # Create a db if there not exists one.
    db_df = pd.read_csv("database.csv", header=0, encoding="utf-8")
    print("Opened database successfully")
    print(db_df.head(5))

    conn = sqlite3.connect("fake.db")
    c = conn.cursor()

    # Create a table KB to save KB triples.
    c.execute("""CREATE TABLE TABLE1
    (ID INT PRIMARY KEY NOT NULL,
    NUM TEXT NOT NULL,
    TEL TEXT NOT NULL,
    CODE TEXT NOT NULL,
    FMS TEXT NOT NULL,
    PROGRESS TEXT NOT NULL);""")
    print("Table created successfully")

    # Insert KB triples into KB table.
    for _ID, _NUM, _TEL, _CODE, _FMS, _PROGRESS in zip(db_df.index, db_df.NUM, db_df.TEL, db_df.CODE, db_df.FMS, db_df.PROGRESS):
        tmp = (_ID, _NUM, _TEL, _CODE, _FMS, _PROGRESS, )
        # print(type(_NUM), type(_TEL), type(_CODE), type(_FMS), type(_PROGRESS))
        # print(tmp)
        c.execute("INSERT INTO TABLE1(ID, NUM, TEL, CODE, FMS, PROGRESS) \
            VALUES(?,?,?,?,?,?)", tmp)

    conn.commit()
    conn.close()


# def select(c, num=None, tel=None, code=None, fms=None):
#     s = "SELECT * FROM TABLE1 WHERE {}=? AND {}=? AND {}=? AND {}=?".format()
#     # c.execute("SELECT * FROM TABLE1 WHERE NUM=? AND TEL=? AND CODE=? AND FMS=?", (num, tel, code, fms,))
#     res = c.fetchall()
#     print(res)

#
# def select(c, slots):
#
#     slots_keys = list(slots.keys())
#     slots_values = list(slots.values())
#
#     if len(slots) == 1:
#         s = "SELECT * FROM TABLE1 WHERE {}=?".format(slots_keys[0])
#         c.execute(s, (slots_values[0],))
#
#     elif len(slots) == 2:
#         s = "SELECT * FROM TABLE1 WHERE {}=? AND {}=?".format(slots_keys[0], slots_keys[1])
#         c.execute(s, (slots_values[0], slots_values[1],))
#
#     elif len(slots) == 3:
#         s = "SELECT * FROM TABLE1 WHERE {}=? AND {}=? AND {}=?".format(slots_keys[0], slots_keys[1], slots_keys[2])
#         c.execute(s, (slots_values[0], slots_values[1], slots_values[2]))
#
#     elif len(slots) == 4:
#         s = "SELECT * FROM TABLE1 WHERE {}=? AND {}=? AND {}=? AND {}=?".format(
#             slots_keys[0], slots_keys[1], slots_keys[2], slots_keys[3])
#         c.execute(s, (slots_values[0], slots_values[1], slots_values[2], slots_values[3]))
#
#     res = c.fetchall()
#     return res
#
#
# slots_dict = {"NUM": 2658375, "TEL": 15729201272}
# select(c1, slots_dict)

def select(query):

    num = str(query['NUM']) if query['NUM'] != '' else None
    tel = str(query['TEL']) if query['TEL'] != '' else None
    code = str(query['CODE']) if query['CODE'] != '' else None
    fms = str(query['FMS']) if query['FMS'] != '' else None

    with closing(sqlite3.connect("fake.db")) as conn:
        c = conn.cursor()

        s = []
        for value, label in zip((num, tel, code, fms), ("NUM", "TEL", "CODE", "FMS")):
            if value:
                s.append("{}='{}'".format(label, value))
        sql = " AND ".join(s)
        sql = "SELECT * FROM TABLE1 WHERE " + sql
        c.execute(sql)
        res = c.fetchall()
        return res


if __name__ == "__main__":
    create_db()
    print(select(query={"NUM": "", "TEL": "13387598270", "CODE": "", "FMS": ""}))


