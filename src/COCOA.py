import numpy as np
import vertica_python
from operator import itemgetter
import pandas as pd
import math
import re


def get_dataset(file_name, use_default_path = True):
    base_url = '../datasets/'
    if use_default_path:
        file = pd.read_csv(base_url+file_name+'.csv', sep=',')
    else:
        file = pd.read_csv(file_name+'.csv', sep=',')
    file = file.apply(lambda x: x.astype(str).str.lower())
    return file


def get_cleaned_text(text):
    if text is None:
        return ''
    stopwords = ['a','the','of','on','in','an','and','is','at','are','as','be','but','by','for','it','no','not','or'
        ,'such','that','their','there','these','to','was','with','they','will',  'v', 've', 'd']

    cleaned = re.sub('[\W_]+', ' ', text.encode('ascii', 'ignore').decode('ascii'))
    feature_one = re.sub(' +', ' ', cleaned).strip()

    for x in stopwords:
        feature_one = feature_one.replace(' {} '.format(x), ' ')
        if feature_one.startswith('{} '.format(x)):
            feature_one = feature_one[len('{} '.format(x)):]
        if feature_one.endswith(' {}'.format(x)):
            feature_one = feature_one[:-len(' {}'.format(x))]
    return feature_one


def generate_rank(column):
    return column.rank()


def generate_join_map(query_column, dict):
    vals = dict.values()
    vals = [int(x) for x in vals]
    join_table = np.full(max(vals) + 1, -1)

    q = np.array(query_column)
    for i in np.arange(len(q)):
        x = q[i]
        index = dict.get(x, -1)
        if index != -1:
            join_table[index] = i

    return join_table


def get_overlappings(k, file_path, query_column):
    conn_info = {'host': '127.0.0.1',
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
    data = get_dataset(file_path)[[query_column]]
    data[query_column] = data[query_column].apply(get_cleaned_text)
    distinct_clean_values = data[query_column].unique()
    joint_distinct_values = '\',\''.join(distinct_clean_values).encode('utf-8')
    
    query = 'SELECT SUBQ.ids FROM (SELECT table_col_id AS ids,' \
            'CONCAT(table_col_id,CONCAT(\'_\',REGEXP_REPLACE(REGEXP_REPLACE(' \
            'tokenized, \'\W+\', \' \'), \' +\', \' \'))) AS COL_ELEM from cbi_inverted_index_2 WHERE REGEXP_REPLACE(' \
            'REGEXP_REPLACE(tokenized, \'\W+\', \' \'), \' +\', \' \') IN (\'{}\') ' \
            'GROUP BY table_col_id,CONCAT(table_col_id,CONCAT(\'_\',' \
            'REGEXP_REPLACE(REGEXP_REPLACE(tokenized, \'\W+\', \' \'), \' +\', \' \'))) ) AS SUBQ GROUP BY SUBQ.ids ' \
            'HAVING COUNT(COL_ELEM) > {} ' \
            'ORDER BY COUNT(COL_ELEM) DESC LIMIT {};'.format(joint_distinct_values, 3, k)
    cur.execute(query)
    result = [item for sublist in cur.fetchall() for item in sublist]
    return result


def enrich_COCOA(dataset_name, data_path, query_column, target_column, k_c, k_t):
    conn_info = {'host': '127.0.0.1',
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

    table_ids = []
    column_ids = []

    overlappings = get_overlappings(k_t, dataset_name, query_column)

    for o in overlappings:
        table_ids.append(int(o.split('_')[0].strip()))
        column_ids.append(int(o.split('_')[1].strip()))

    s = [str(i) for i in table_ids]

    cur.execute('SELECT tableid, Max(colid) from tbl_inverted_index WHERE tableid IN (\'{}\') GROUP BY tableid;'.format('\',\''.join(s)))
    result = pd.DataFrame(cur.fetchall(), columns=['tableid', 'max_col_id'])
    max_column_dict = result.set_index('tableid').to_dict()['max_col_id']

    data = get_dataset(data_path, use_default_path=False)[[query_column] + [target_column]]
    data[query_column] = data[query_column].apply(lambda x: get_cleaned_text(x))

    data['rank_target'] = generate_rank(data[target_column])
    data['rank_target'] = data['rank_target'].astype('int64')
    input_size = len(data)

    column_name = []
    column_correlation = []
    column_content = []

    conditions = []
    for i in np.arange(len(table_ids)):
        conditions += ['{}_{}'.format(table_ids[i], column_ids[i])]
    token_query = 'SELECT table_col_id as first_level_key, tokenized, rowid FROM tbl_inverted_index WHERE table_col_id IN (\'{}\') order by tableid, colid, rowid;'.format(
        '\',\''.join(conditions))
    cur.execute(token_query)
    external_joinable_tables = pd.DataFrame(cur.fetchall(), columns=['first_level_key', 'tokenized', 'rowid'])

    groups = external_joinable_tables.groupby(['first_level_key'])
    joinable_tables_dict = {}
    for name, group in groups:
        keys = list(group['tokenized'])
        values = list(group['rowid'])
        item = dict(zip(keys, values))
        joinable_tables_dict[name] = item

    order_query = 'SELECT order_index, binary_list, min_index, CONCAT(CONCAT(tableid, \'_\'), colid) FROM tbl_order_index WHERE tableid IN (\'{}\');'.format(
        '\',\''.join(s))
    cur.execute(order_query)

    order_dict = {}
    binary_dict = {}
    min_dict = {}
    orders_df = pd.DataFrame(cur.fetchall(), columns=['order', 'binary', 'min_index', 'table_col_id']).groupby(['table_col_id'])
    for name, group in orders_df:
        order_dict[name] = list(group['order'])[0]
        binary_dict[name] = list(group['binary'])[0]
        min_dict[name] = list(group['min_index'])[0]

    content_query = 'SELECT table_col_id, tokenized FROM cbi_inverted_index WHERE tableid IN (\'{}\') order by tableid, colid, rowid;'.format(
        '\',\''.join(s))
    cur.execute(content_query)
    content_dict = {}
    content_df = pd.DataFrame(cur.fetchall(), columns=['table_col_id', 'tokenized']).groupby(['table_col_id'])
    for name, group in content_df:
        content_dict[name] = list(group['tokenized'])

    numerics_dict = {}
    with open("../{}_floats_{}_{}.txt".format(dataset_name, k_t, input_size), "r") as f:
        for line in f:
            key = '{}_{}'.format(line.split('_')[0].strip(), line.split('_')[1].strip())
            if key in numerics_dict:
                numerics_dict[key] = numerics_dict[key] + [int(line.split('_')[2].strip())]
            else:
                numerics_dict[key] = [int(line.split('_')[2].strip())]

    table_number = len(table_ids)
    for i in np.arange(table_number):
        column = column_ids[i]
        table = table_ids[i]
        max_col = max_column_dict[table]

        joinMap = generate_join_map(data[query_column], joinable_tables_dict[str(table) + '_' + str(column)])

        for c in np.arange(max_col + 1):
            if '{}_{}'.format(table, column) not in numerics_dict or c not in numerics_dict['{}_{}'.format(table, column)]:
                continue
            if c == column:
                continue

            data['new_external_rank'] = math.ceil(input_size / 2)
            external_rank = data['new_external_rank'].values
            data['new_external_content'] = ""
            external_content = data['new_external_content'].values

            order_index = order_dict['{}_{}'.format(table, c)]
            binary_index = binary_dict['{}_{}'.format(table, c)]
            starting_point = min_dict['{}_{}'.format(table, c)]
            content_list = np.array(list(content_dict['{}_{}'.format(table, c)]))

            order = order_index.split(', [')[1][:-2].split(', ')
            order = [int(x) for x in order]

            binary = binary_index.split(', [')[1][:-2].split(', ')
            pointer = starting_point

            assignment_flag = False
            counter = 1
            skipped = 1
            while pointer != -1:
                input_index = joinMap[pointer]
                if input_index != -1:
                    external_rank[input_index] = counter
                    assignment_flag = True
                    external_content[input_index] = str(content_list[pointer])
                if binary[pointer] == 'T':
                    if assignment_flag:
                        counter += skipped
                        skipped = 1
                        assignment_flag = False
                    else:
                        skipped += 1
                next = order[pointer]
                pointer = next

            cor = np.corrcoef(data['rank_target'], external_rank)[0, 1]
            column_name += [str(table) + '_' + str(c)]
            column_correlation += [cor]
            column_content += [external_content.copy()]

    overall_list = []
    for i in np.arange(len(column_correlation)):
        overall_list += [[column_correlation[i], column_name[i], column_content[i]]]
    sorted_list = sorted(overall_list, key=itemgetter(0), reverse=True)
    for important_column_index in np.arange(min(k_c, len(sorted_list))):
        important_column = sorted_list[important_column_index]
        data['{}_{}'.format(important_column[1], important_column_index)] = important_column[2]
    data = data.drop(['new_external_rank', 'new_external_content'], axis=1)
    connection.close()
    return data


