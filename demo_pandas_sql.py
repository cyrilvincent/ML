import sqlite3
import pandas as pd

# PostgreSql psycopg2
# SqlServer pymssql
# Oracle python-oracledb

connection = sqlite3.connect("data/house/house.db3")
print(connection)

dataframe = pd.read_sql("select * from house", connection)
print(dataframe)
