# !pip install google-cloud-bigquery google-auth db-dtypes pandas numpy openai
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import os, openai
from config import config
from typing import Union, List
from openai import OpenAI

def get_bigquery_client(type: str) -> Union[bigquery.Client, None]:
    """
    Get a BigQuery client based on the specified type.

    Args:
        type (str): The type of BigQuery client to retrieve. Valid values are "job" and "work".

    Returns:
        Union[bigquery.Client, None]: The BigQuery client object if successful, None otherwise.
    """
    try:
        if type == "job":
            path = config.get("bigquery_client_jobmarket_service_account_path")
            scopes = config.get("bigquery_client_jobmarket_scopes")
            project = config.get("bigquery_client_jobmarket_project")
        elif type == "work":
            path = config.get("bigquery_client_work_service_account_path")
            scopes = config.get("bigquery_client_work_scopes")
            project = config.get("bigquery_client_work_project")
        else:
            raise ValueError("Invalid type")

        if not all([path, scopes, project]):
            raise ValueError("Missing or invalid configuration")

        credentials = service_account.Credentials.from_service_account_file(path, scopes=scopes)
        return bigquery.Client(credentials=credentials, project=project)
    except Exception as e:
        print(f"Error: Cannot get BigQuery client: {str(e)}")
        return None

def get_bigquery_table(dataset: str, table_name: str, client: bigquery.Client) -> pd.DataFrame:
    """
    Retrieves a BigQuery table and returns it as a pandas DataFrame.

    Args:
        dataset (str): The name of the dataset containing the table.
        table_name (str): The name of the table to retrieve.
        client (bigquery.Client): The BigQuery client.

    Returns:
        pd.DataFrame: The table data as a pandas DataFrame.
    """
    project = client.project
    table_ref = bigquery.DatasetReference(project, dataset).table(table_name)
    table = client.get_table(table_ref)
    
    return client.list_rows(table).to_dataframe()

def get_bigquery_query(query: str, client: bigquery.Client) -> pd.DataFrame:
    """
    Executes a BigQuery query using the provided client and returns the result as a pandas DataFrame.

    Args:
        query (str): The BigQuery query to execute.
        client (bigquery.Client): The BigQuery client to use for executing the query.

    Returns:
        pd.DataFrame: The result of the query as a pandas DataFrame.
    """
    return client.query(query).to_dataframe()

def get_bigquery_query(query: str, client: bigquery.Client) -> pd.DataFrame:
    """
    Executes a BigQuery query using the provided client and returns the result as a pandas DataFrame.

    Args:
        query (str): The BigQuery query to execute.
        client (bigquery.Client): The BigQuery client to use for executing the query.

    Returns:
        pd.DataFrame: The result of the query as a pandas DataFrame.
    """
    return client.query(query).to_dataframe()
def get_bigquery_query(query: str, client: bigquery.Client):
    
    return client.query(query).to_dataframe()

def get_splited_list(x: list, n: int):
    """
    Splits a list into sublists of size n.
    
    Args:
        x (list): The input list to be split.
        n (int): The size of each sublist.
    
    Returns:
        list: A list of sublists, each containing n elements.
    """
    return [x[i:i+n] for i in range(0, len(x), n)]

def get_openai_embedding(x: List[str], key_type: str = "personal") -> List[float]:
    """
    Get OpenAI embeddings for a list of texts.

    Args:
        x (List[str]): The list of texts to be embedded.
        key_type (str, optional): The type of API key to use. Defaults to "personal".

    Returns:
        List[float]: The embeddings for the input texts.
    Raises:
        ValueError: If an invalid key_type is provided.
    """
    if key_type == "personal":
        api_key = config.get("open_ai_key_personal")
    elif key_type == "work":
        api_key = config.get("open_ai_key_work")
    else:
        raise ValueError("Invalid type")
    
    openai_client = OpenAI(api_key=api_key)
    
    return openai_client.embeddings.create(input=x, model="text-embedding-ada-002").data
