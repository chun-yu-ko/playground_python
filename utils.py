# !pip install google-cloud-bigquery google-auth db-dtypes pandas numpy openai
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
import os, openai, subprocess, requests, replicate
from config import config
from typing import Union, List
from openai import OpenAI
import plotly.express as px
from sklearn.metrics import confusion_matrix
from ucimlrepo import fetch_ucirepo

subprocess.run(["huggingface-cli", "login", "--token", config.get("huggingface_token")])
os.environ["REPLICATE_API_TOKEN"] = config.get("replicate_token")

px.set_mapbox_access_token(config.get("mapbox_token"))



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

def print_binary_classifier_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics based on the confusion matrix.

    Args:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels

    Returns:
    - metrics: dictionary, evaluation metrics
    """

    cm = confusion_matrix(y_true, y_pred)

    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    npv = TN / (TN + FN)
    fpr = FP / (FP + TN)
    fdr = FP / (FP + TP)
    fnr = FN / (FN + TP)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    metrics = {
        "Confusion Matrix": pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"]),
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Negative Predictive Value": npv,
        "False Positive Rate": fpr,
        "False Discovery Rate": fdr,
        "False Negative Rate": fnr,
        "F1 Score": f1_score,
        "Matthews Correlation Coefficient": mcc
    }
    
    print("Evaluation Metrics:")
    
    for key, value in metrics.items():
        print(key + ":")
        print(value)
        print("--------------------")

def get_answer_gimini(x):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": config.get("gimini_token")}
    json_data = {"contents": [{"parts": [{"text": x}]}], "generationConfig": {"temperature": 0}}
    response = requests.post(url, params = params, headers = headers, json = json_data)
    return response

def get_answer_llama(x):
    response = replicate.run("meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3", input={"debug": False, "top_k": 50, "top_p": 1, "prompt": x, "temperature": 0.01, "system_prompt": "", "max_new_tokens": 500, "min_new_tokens": -1})
    return response

def get_answer_chatgpt(x):
    openai_client = OpenAI(api_key = config.get("open_ai_key_personal"))
    response = openai_client.chat.completions.create(model = "gpt-4-1106-preview", messages = [{"role": "user", "content": x}], temperature = 0)
    return response

def get_answer_mistral(x):
    response = replicate.run("mistralai/mistral-7b-v0.1:3ab1d218053dab642ffd4608fcaa4a864cae1d4a9c5dde7b7db939e8af3767af", input={"debug": False, "top_k": 50, "top_p": 1, "prompt": x, "temperature": 0.01, "system_prompt": "", "max_new_tokens": 500, "min_new_tokens": -1})
    return response

def get_data_from_ucirepo(x):
    # Fetch data from UCIR repo
    ori_data = fetch_ucirepo(id=x)
    
    # Concatenate features and targets
    data = pd.concat([ori_data.data.features, ori_data.data.targets], axis=1)
    
    # One-hot encode categorical variables
    for i in ori_data.variables.query("type == 'Categorical'").name.tolist():
        data = pd.concat([data, pd.get_dummies(data[i], prefix=i, drop_first=True).astype(int)], axis=1)
        data.drop(i, axis=1, inplace=True)
    
    return data