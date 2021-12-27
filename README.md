# COCOA
### COrrelation COefficient-Aware Data Augmentation

## Table of Contents
  - [Prerequisites](#prerequisites)
    - [Python version and packages](#python-version-and-packages)
    - [Database schemas](#database-schemas)
  - [Installation](#installation)
  - [Usage](#usage)

## Prerequisites
### Python version and packages
This packages requires python version >= 3.6. Additionally, the following packages have to be installed:
```
pandas
numpy
```

### Database schemas
To run the system, the following set of tables need to be created in the database (e.g. Postgres):
- ```main_tokenized```: Inverted index, tokenized -> table, col, row
- ```distinct_tokens```: Like ```main_tokenized```, but maps only distinct tokenized -> table_col_id to compute
overlap faster
- ```order_index```: Index for each column. The index is created offline and stored in the db.
- ```max_column```: Maps table-> number of columns for quicker access

Schemas:
```sql
CREATE TABLE main_tokenized (
    tokenized TEXT,
    tableid INT NOT NULL,
    rowid INT NOT NULL,
    table_col_id TEXT NOT NULL,
);

CREATE TABLE distinct_tokens (
    tokenized TEXT, 
    table_col_id TEXT NOT NULL
);

CREATE TABLE order_index (
    table_col_id TEXT NOT NULL
    is_numeric BOOLEAN,
    min_index INT NOT NULL,
    order_list TEXT,
    binary_list TEXT,
);

CREATE TABLE max_column (
    tableid INT NOT NULL,
    max_col_id INT NOT NULL,
    PRIMARY KEY (tableid)
);
```

To fill ```distinct_tokens``` and ```max_column```, run the following queries:
```sql
INSERT INTO distinct_tokens
SELECT DISTINCT tokenized, table_col_id
FROM main_tokenized;

INSERT INTO max_column 
SELECT tableid, MAX(colid)
FROM main_tokenized
GROUP BY tableid;
```

The order index can be created by calling
```python
COCOA.create_index(values)
```
where values is a column (list of values). The index then has to be stored in the ```order_index``` table in the db for 
every column in the dataset.

## Installation
Run the following command to install COCOA to your python environment:
```
pip install cocoa-system
```

## Usage
```python
import pandas as pd
import psycopg2
from cocoa_system import DataAugmentation

CONN_INFO = {
    'host': '127.0.0.1',
    'dbname': 'db',
    'user': 'postgres',
    'password': 'password',
}

DB_TABLES = {
    'dt': 'distinct_tokens',
    'mt': 'main_tokenized_overlap',
    'mc': 'max_column',
    'oi': 'order_index',
}

DATASET_PATH = 'datasets/query.csv'
dataset = pd.read_csv(DATASET_PATH)

conn = psycopg2.connect(**CONN_INFO)
cocoa = DataAugmentation.COCOAHandler(conn, DB_TABLES)
result = cocoa.enrich(dataset, k_c, k_t, 'my_query_column', 'my_target_column')
```