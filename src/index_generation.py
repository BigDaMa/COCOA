import vertica_python
import numpy as np
import scipy.stats as ss
import time


def generate_inverted_index():
    conn_info = {'host': 'SERVER_IP_ADDRESS',
                 'port': 5433,
                 'user': 'USERNAME',
                 'password': 'PASSWORD',
                 'database': 'DATABASE_NAME',
                 'session_label': 'some_label',
                 'read_timeout': 6000,
                 'unicode_error': 'strict',
                 }
    connection = vertica_python.connect(**conn_info)
    cur = connection.cursor()

    table_ids_query = 'CREATE TABLE tbl_inverted_index DIRECT AS SELECT /*+DIRECT*/ tokenized, LISTAGG (tableid||colid||rowid) from main_tokenized_union;'
    cur.execute(table_ids_query)


def is_numeric_list(l):
    for i in np.arange(len(l)):
        if l[i] is None:
            l[i] = np.nan
        else:
            l[i] = l[i]
    result = [s for s in l if is_number(str(s))]
    return len(result) == len(l)


def generate_order_index():
    conn_info = {'host': 'SERVER_IP_ADDRESS',
                 'port': 5433,
                 'user': 'USERNAME',
                 'password': 'PASSWORD',
                 'database': 'DATABASE_NAME',
                 'session_label': 'some_label',
                 'read_timeout': 6000,
                 'unicode_error': 'strict',
                 }
    connection = vertica_python.connect(**conn_info)
    cur = connection.cursor()
    table_column_ids_query = 'select tableid, max(colid) from main_tokenized_union group by tableid;'
    cur.execute(table_column_ids_query)
    table_column_ids = cur.fetchall()
    for table_column_id in table_column_ids:
        table_id = table_column_id[0]
        print('table {} is indexing'.format(table_id))
        max_col_id = table_column_id[1]

        for c in np.arange(0, max_col_id + 1, 1):
            column_content_query = 'select rowid, tokenized from main_tokenized_union where ' \
                                   'tableid = {} and colid = {} order by rowid'.format(table_id, c)
            cur.execute(column_content_query)
            column_content = cur.fetchall()  # token

            if is_numeric_list([item[1] for item in column_content]): #checks if the column is numeric
                for i in np.arange(len(column_content)):
                    if column_content[i][1] is None:
                        column_content[i][1] = np.nan
                rows = [i[0] for i in column_content]
                values = [float(i[1]) for i in column_content]
                ranks = ss.rankdata(values)

                rows_sorted_based_on_ranks = [x for _,x in sorted(zip(ranks,rows))]
                min_index = rows_sorted_based_on_ranks[0] #starting point in the order index
                order_list = np.empty(len(rows), dtype=int)
                binary_list = np.empty(len(rows), dtype=str)
                sorted_ranks = np.sort(ranks).copy()
                for i in np.arange(len(rows)-1):
                    order_list[i] = rows_sorted_based_on_ranks[i+1]
                    if sorted_ranks[i] == sorted_ranks[i+1]:
                        binary_list[i] = 'F'
                    else:
                        binary_list[i] = 'T'
                order_list[len(rows)-1] = -1 #Maximum value
                binary_list[len(rows)-1] = -1 #Maximum value

                final_order_list = [x for _,x in sorted(zip(rows_sorted_based_on_ranks,order_list))] #order list in the order index
                final_binary_list = [x for _,x in sorted(zip(rows_sorted_based_on_ranks,binary_list))] #binary list in the order index

                cur.execute("INSERT INTO tbl_order_index VALUES ({}, {}, {}, \'{}\', \'{}\');".format(
                    table_id, c, min_index, ','.join(str(v) for v in final_order_list), ','.join(str(v) for v in final_binary_list)))
        cur.execute("COMMIT")


generate_order_index()
