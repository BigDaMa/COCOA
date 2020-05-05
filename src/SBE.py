import numpy as np
import vertica_python
from operator import itemgetter
import pandas as pd
from scipy.stats import spearmanr
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


def spearmans_correlation(x, y):
    y2 = np.array(y.fillna(0.0))
    if len(y) < 3 or len(x) < 3:
        return 0.0
    if np.count_nonzero(~np.isnan(x)) < 3 or np.count_nonzero(~np.isnan(y2)) < 3:
        return 0.0
    x2 = np.nan_to_num(x)
    correlation = spearmanr(x2, y2)
    if np.isnan(correlation[0]):
        return 0
    return correlation[0]


def enrich_SBE(dataset_name, query_column, target_column, k_c, k_t):
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

    cur.execute('SELECT tableid, Max(colid) from tbl_inverted_index WHERE tableid IN (\'{}\') GROUP BY ttableid;'.format('\',\''.join(s)))
    result = pd.DataFrame(cur.fetchall(), columns=['tableid', 'max_col_id'])
    max_column_dict = result.set_index('tableid').to_dict()['max_col_id']

    data = get_dataset(dataset_name)[[query_column] + [target_column]]
    data[query_column] = data[query_column].apply(lambda x: get_cleaned_text(x))

    data[target_column] = data[target_column].astype('float')
    input_size = len(data)

    column_name = []
    column_correlation = []
    column_content = []

    tables_fetch_query = 'SELECT tableid, colid, rowid, table_row_id, tokenized FROM tbl_inverted_index WHERE tableid IN (\'{}\') order by tableid, colid, rowid;'.format(
        '\',\''.join(s))
    cur.execute(tables_fetch_query)
    external_tables = pd.DataFrame(cur.fetchall(), columns=['tableid', 'colid', 'rowid', 'table_row_id', 'tokenized'])

    temp = external_tables.sort_values(by=['tableid', 'rowid', 'colid']).groupby(['tableid', 'rowid']).tokenized.apply(
        list).reset_index()

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

        temp_condition = temp['tableid'] == table
        external_temp = temp[temp_condition].drop(['tableid'], axis=1).set_index('rowid').to_dict()['tokenized']
        df_temp = pd.DataFrame.from_dict(external_temp, orient='index').drop_duplicates(column, keep='last')

        df_cd = pd.merge(data, df_temp, how='left', left_on=query_column, right_on=column)

        df_cd.applymap(lambda x: str(x).replace(',', '').replace('$', '').replace('.', '').replace(' ', '').replace(':',
                                                                                                                    '').replace(
            ';', '').replace('%', '').replace('&', '').replace('?', '').replace('/', '')).to_csv(
            'temp_{}_{}.csv'.format(dataset_name, k_t), index=False)
        df_cd = pd.read_csv('temp_{}_{}.csv'.format(dataset_name, k_t))

        for c in range(max_col + 1):
            if '{}_{}'.format(table, column) not in numerics_dict or c not in numerics_dict['{}_{}'.format(table, column)]:
                continue
            if c != column:

                column_to_be_added = df_cd[str(c)]

                cor = abs(spearmans_correlation(data[target_column], column_to_be_added))
                column_name += [str(table) + '_' + str(c)]
                column_correlation += [cor]
                column_content += [column_to_be_added.copy()]

    overall_list = []
    for i in np.arange(len(column_correlation)):
        overall_list += [[column_correlation[i], column_name[i], column_content[i]]]
    sorted_list = sorted(overall_list, key=itemgetter(0), reverse=True)
    for important_column_index in np.arange(min(k_c, len(sorted_list))):
        important_column = sorted_list[important_column_index]
        data['{}_{}'.format(important_column[1], important_column_index)] = important_column[2]

    connection.close()
    return data



