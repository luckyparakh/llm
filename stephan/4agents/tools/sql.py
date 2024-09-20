from langchain.tools import Tool
from pydantic.v1 import BaseModel
from typing import List
import sqlite3
conn=sqlite3.connect("db.sqlite")

def get_tables():
    c=conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables=c.fetchall()
    return "\n".join([table[0] for table in tables if table[0] is not None])
 
def run_query(query):
    c=conn.cursor()
    try:
        c=c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"

# Pydantic BaseModel class is used to define the schema for the arguments of the function
class RunQueryArgs(BaseModel):
    # The query argument is defined as a string
    query: str

run_query=Tool.from_function(
    name="run_sqlite_query",
    description="Run a Sqlite query",
    func=run_query,
    # generate a schema for the arguments of the function using Pydantic BaseModel class
    # it will be used to validate the input arguments to the function, it like struct in GO
    args_schema=RunQueryArgs
)
class DescribeTablesArgs(BaseModel):
    # The table_names argument is defined as a list of strings
    table_names: List[str]

def describe_tables(table_names):
    c=conn.cursor()
    tables=", ".join([f"'{table_name}'" for table_name in table_names])
    rows=c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables})")
    return "\n".join([row[0] for row in rows if row[0] is not None])

describe_tables=Tool.from_function(
    name="describe_tables",
    description="Given list of table names, returns the schema of a table",
    func=describe_tables,
    args_schema=DescribeTablesArgs
)