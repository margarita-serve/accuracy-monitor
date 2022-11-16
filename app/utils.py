import os
import base64
import shutil
import logging
import time

from minio import Minio
from elasticsearch import Elasticsearch, exceptions, helpers
from secrets import token_hex
from functools import lru_cache
from datetime import datetime
import pandas as pd
from tensorflow import keras
import torch
import pickle
import joblib
import lightgbm
import xgboost

KSERVE_API_DEFAULT_STORAGE_ENDPOINT = os.environ.get('KSERVE_API_DEFAULT_STORAGE_ENDPOINT')
KSERVE_API_DEFAULT_DATABASE_ENDPOINT = os.environ.get('KSERVE_API_DEFAULT_DATABASE_ENDPOINT')
KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID = os.environ.get('KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID')
KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY = os.environ.get('KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY')

ES = Elasticsearch(KSERVE_API_DEFAULT_DATABASE_ENDPOINT)

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def split_path(data):
    bucket_name = data[:data.find('/')]
    path = data[data.find('/') + 1:]

    return [bucket_name, path]


def decode_key(value):
    b_bytes = value.encode('ascii')
    m_bytes = base64.b64decode(b_bytes)
    decode_value = m_bytes.decode('ascii')
    return decode_value


def minio_client(path):
    sp = split_path(path)
    access_key = decode_key(KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID)
    secret_key = decode_key(KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY)
    client = Minio(KSERVE_API_DEFAULT_STORAGE_ENDPOINT, access_key, secret_key, secure=False)

    path_token = token_hex(8)
    for item in client.list_objects(sp[0], recursive=True):
        if sp[1] in item.object_name:
            client.fget_object(sp[0], item.object_name, f"temp/{path_token}/{item.object_name}")
    path = f"temp/{path_token}/{sp[1]}"
    return path


def delete_file(path):
    paths = path.split('/')
    try:
        shutil.rmtree(f"{paths[0]}/{paths[1]}")
    except Exception as err:
        logger.exception(err)


class LoadData:
    def __init__(self, index, inference_id):
        self.index = index
        self.inference_id = inference_id

    @lru_cache(maxsize=32)
    def load_data(self):
        try:
            value = ES.get(index=self.index, id=self.inference_id, request_timeout=30)['_source']
            return True, value

        except exceptions.ConnectionTimeout as err:
            logger.exception(err)
            return False, err

        except Exception as err:
            logger.exception(err)
            return False, err


def save_data(index, name, document):
    make_index(index)
    try:
        ES.index(index=index, id=name, document=document)
        logger.info(f"Success input data in Index: {index} ID: {name}")
        return True, f"Success input data in Index: {index} ID: {name}"

    except Exception as err:
        logger.exception(err)
        return False, err


def make_index(index_name):
    if not ES.indices.exists(index=index_name):
        mapping = {"settings": {'mapping': {'ignore_malformed': True}}}
        ES.indices.create(index=index_name, body=mapping)
        logger.info(f"Success create Index : {index_name}")


def create_indices(index):
    try:
        ES.indices.create(index=index)
        return f'Create index: {index}'
    except exceptions.NotFoundError as err:
        return err
    except Exception as err:
        return err


def msearch_query(query):
    try:
        res = ES.msearch(
            body=query
        )
        return True, res
    except Exception as err:
        return False, err


def upsert_doc(index, name, doc):
    make_index(index)
    try:
        ES.update(index=index, id=name, body={"doc": doc, 'upsert': doc})
        logger.info(f'index : {index} id : {name} Success Upsert')
        return True, f'index : {index} id : {name} Success Upsert'
    except Exception as err:
        # logger.exception(err)
        return False, err


def update_data(index, name, document):
    try:
        ES.update(index=index, id=name, body={'doc': document})
        logger.info(f'index : {index} id : {name} Success Update')
        return True, f'index : {index} id : {name} Success Update'
    except Exception as err:
        return False, err


def es_bulk(bulk):
    try:
        result = helpers.bulk(ES, bulk)
        logger.info(f'Success bulk post. count: {result[0]}')
        return True, 'Success bulk post', result[0]
    except Exception as err:
        logger.exception('bulk err')
        return False, err, 0


def search_index(index, query):
    try:
        items = ES.search(index=index, query=query, scroll='30s', size=100)
        sid = items['_scroll_id']
        fetched = items['hits']['hits']
        total = []

        for i in fetched:
            total.append(i)
        while len(fetched) > 0:
            items = ES.scroll(scroll_id=sid, scroll='30s')
            fetched = items['hits']['hits']
            for i in fetched:
                total.append(i)
            time.sleep(0.001)

        return True, total
    except exceptions.NotFoundError as err:
        return False, err
    except Exception as err:
        return False, err


def search_data(index, start_time, end_time, aggs):
    query = get_query(start_time, end_time)
    try:
        items = ES.search(index=index, query=query, aggs=aggs, size=10)
        if items['hits']['total']['value'] == 0:
            raise Exception(f'No data between {start_time} ~ {end_time} ')
        else:
            value = items['aggregations']
        return True, value
    except exceptions.NotFoundError as err:
        return False, err
    except Exception as err:
        logger.exception(f'search data func Error : {err}\n')
        return False, err


def get_count_inference(index, start_time, end_time, aggs):
    query = {
        "range": {
            "timestamp": {
                "gte": f"{start_time}",
                "lt": f"{end_time}"
            }
        }
    }
    try:
        items = ES.search(index=index, query=query, aggs=aggs, size=0)
        if items['hits']['total']['value'] == 0:
            return False, f'No data between {start_time} ~ {end_time} '
        else:
            return True, items['aggregations']
    except exceptions.NotFoundError as err:
        return False, err
    except Exception as err:
        logger.exception(f'search data func Error : {err}\n')
        return False, err


def search_all_data(index, aggs):
    try:
        items = ES.search(index=index, aggs=aggs, size=10)
        value = items['aggregations']
        return True, value
    except exceptions.NotFoundError as err:
        return False, err
    except Exception as err:
        logger.exception(f'search data func Error : {err}\n')
        return False, err


def get_query(start_time, end_time):
    q = {
        "range": {
            "date": {
                "gte": f"{start_time}",
                "lt": f"{end_time}"
            }
        }
    }
    return q


def get_aggs(model_type):
    if model_type == 'Regression':
        aggs = {
            "average_predicted": {
                "sum": {
                    "field": "average_predicted"
                }
            },
            "average_actual": {
                "sum": {
                    "field": "average_actual"
                }
            },
            "count": {
                "sum": {
                    "field": "count"
                }
            },
            "rmse": {
                "sum": {
                    "field": "rmse"
                }
            },
            "rmsle": {
                "sum": {
                    "field": "rmsle"
                }
            },
            "mae": {
                "sum": {
                    "field": "mae"
                }
            },
            # "median_absolute_error": {
            #     "sum": {
            #         "field": "median_absolute_error"
            #     }
            # },
            "mape": {
                "sum": {
                    "field": "mape"
                }
            },
            "mean_tweedie_deviance": {
                "sum": {
                    "field": "mean_tweedie_deviance"
                }
            },
            "mean_gamma_deviance": {
                "sum": {
                    "field": "mean_gamma_deviance"
                }
            }
        }
    elif model_type == 'Binary':
        aggs = {
            "percentage_predicted_count": {
                "sum": {
                    "field": "percentage_predicted"
                }
            },
            "percentage_actual_count": {
                "sum": {
                    "field": "percentage_actual"
                }
            },
            "count": {
                "sum": {
                    "field": "count"
                }
            },
            "tp": {
                "sum": {
                    "field": "tp"
                }
            },
            "fn": {
                "sum": {
                    "field": "fn"
                }
            },
            "fp": {
                "sum": {
                    "field": "fp"
                }
            },
            "tn": {
                "sum": {
                    "field": "tn"
                }
            },
        }
    else:
        aggs = {}

    return aggs


def get_avg_aggs(model_type):
    if model_type == "Regression":
        aggs = {
            "average_predicted": {
                "sum": {
                    "field": "average_predicted"
                }
            },
            "average_actual": {
                "sum": {
                    "field": "average_actual"
                }
            },
            "count": {
                "sum": {
                    "field": "count"
                }
            }

        }
    elif model_type == "Binary":
        aggs = {
            "percentage_predicted": {
                "sum": {
                    "field": "percentage_predicted"
                }
            },
            "percentage_actual": {
                "sum": {
                    "field": "percentage_actual"
                }
            },
            "count": {
                "sum": {
                    "field": "count"
                }
            }

        }
    else:
        aggs = {}

    return aggs


def read_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    if "Unnamed: 0" in df.columns:
        df.pop("Unnamed: 0")

    delete_file(dataset_path)
    logger.info(f"delete {dataset_path}")
    return df


def get_prediction(df, model, framework):
    df_length = len(df)
    slice_value = 50000 if df_length > 50000 else df_length
    df_sample = df.sample(n=slice_value, random_state=1004)
    pred = get_pred(model, df_sample, framework)

    return pred


def load_model(framework, path):
    try:
        if framework == 'TensorFlow':
            model = keras.models.load_model(path)
        elif framework == 'PyTorch':
            model = torch.load(f=path)
        elif framework == 'SkLearn':
            model = joblib.load(path)
        elif framework == 'XGBoost':
            model = xgboost.Booster(model_file=path)
        else:
            model = lightgbm.Booster(model_file=path)
    except Exception:
        try:
            model = pickle.load(open(path, 'rb'))
        except Exception:
            delete_file(path)
            logger.info(f"delete {path}")
            return False

    delete_file(path)
    logger.info(f"delete {path}")
    return model


def get_pred(model, df, framework):
    try:
        if framework == 'PyTorch':
            df_tensor = torch.from_numpy(df.values)
            df_tensor = torch.tensor(df_tensor, dtype=torch.float32)
            pred = model(df_tensor)
            pred = pred.detach().numpy()
        else:
            pred = list(model.predict(df))
    except Exception:
        return False

    return pred


def check_index(inference_name, current_model):
    try:
        status = ES.indices.exists(index=f"{inference_name}_{current_model}")
        return True, status
    except Exception as err:
        return False, err


def convertTimestamp(timeString):
    times = datetime.strptime(timeString, "%Y-%m-%d:%H")
    convTime = times.strftime("%Y-%m-%dT%H:%M:%SZ")

    return convTime


def validate(model, df, target, framwork):
    # 1. 데이터 프레임 타입확인
    for i in df.dtypes:
        if i == float or i == int:
            pass
        else:
            # todo 추후 string 등 추가 예정 현재는 int 와 float만 받음
            return False, "DataSet Type Error : Column data type must be int or float."
    # 2. None type check
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        return False, "DataSet Value Error : The value of the dataset has null."
    # 3. target column check
    if target in df.columns:
        df = df.drop(columns=target)
    else:
        return False, f"DataSet Column Error : Target label :{target} does not exist in dataset column."
    # 4. sample predict
    sample = df.sample(10)
    try:
        if framwork == "PyTorch":
            df_tensor = torch.from_numpy(sample.values)
            df_tensor = torch.tensor(df_tensor, dtype=torch.float32)
            model(df_tensor)
        else:
            model.predict(sample)
    except Exception as err:
        return False, f"Model Predict Error : {err}"

    logger.info("Validate Check Pass")
    return True, "Validate Check Pass"


def validate_actual(df, target, association_column, positive_class=False, negative_class=False):
    column_list = df.columns.to_list()
    # 1. target, association_column in check
    if target in column_list and association_column in column_list:
        pass
    elif target not in column_list and association_column not in column_list:
        logger.critical(
            f"Actual Dataset Value Error : Columns not in target : {target}, association_column : {association_column}")
        return False, f"Actual Dataset Value Error : Columns not in target : {target}, association_column : {association_column}"
    elif target not in column_list:
        logger.critical(
            f"Actual Dataset Value Error : Columns not in target : {target}")
        return False, f"Actual Dataset Value Error : Columns not in target : {target}"
    else:
        logger.critical(
            f"Actual Dataset Value Error : Columns not in association_column : {association_column}")
        return False, f"Actual Dataset Value Error : Columns not in association_column : {association_column}"
    label_df = df[target]
    asso_df = df[association_column]

    # 2. None value check
    target_null_count = label_df.isnull().sum()
    asso_df_null_count = asso_df.isnull().sum()
    if asso_df_null_count > 0 or target_null_count > 0:
        logger.critical("Actual Dataset Value Error : The value of the dataset has null.")
        return False, "Actual Dataset Value Error : The value of the dataset has null."
    # 3. target column check
    if positive_class is not False:
        label_list = label_df.unique().tolist()
        classes = ["1", "0"]
        for i in label_list:
            if str(i) in classes:
                pass
            else:
                logger.critical(f"Actual Dataset Value Error : The value of the target column is not in [1, 0]")
                return False, f"Actual Dataset Value Error : The value of the target column is not in [1, 0]"
    else:
        if label_df.dtype == float or label_df.dtype == int:
            pass
        else:
            # todo 추후 string 등 추가 예정 현재는 int 와 float만 받음
            logger.critical("DataSet Type Error : Regression target column data type must be int or float.")
            return False, "DataSet Type Error : Regression target column data type must be int or float."
    # 4. association column check
    if asso_df.dtype == float or asso_df.dtype == int:
        pass
    else:
        # todo 추후 string 등 추가 예정 현재는 int 와 float만 받음
        logger.critical("DataSet Type Error : Association column data type must be int or float.")
        return False, "DataSet Type Error : Association column data type must be int or float."

    return True, "Validate Check Pass"


def make_percent(base, key, key_value):
    high_list = ['tpr', 'accuracy', 'f1', 'ppv']
    low_list = ['fpr', 'fnr', 'rmse', 'mae', 'mean_gamma_deviance', 'mean_tweedie_deviance', 'rmsle', 'mape']

    if base[key] == 0:
        return 100
    elif base[key] == 'N/A' or key_value == 'N/A':
        return 'N/A'
    else:
        if key in high_list:
            return (key_value - base[key]) / base[key] * 100
        elif key in low_list:
            return (base[key] - key_value) / base[key] * 100
        else:
            return "N/A"


def make_value(base, key, key_value):
    high_list = ['tpr', 'accuracy', 'f1', 'ppv']
    low_list = ['fpr', 'fnr', 'rmse', 'mae', 'mean_gamma_deviance', 'mean_tweedie_deviance', 'rmsle', 'mape']

    if base[key] == 'N/A' or key_value == 'N/A':
        return 'N/A'

    if key in high_list:
        return base[key] - key_value

    elif key in low_list:
        return key_value - base[key]
    else:
        return 'N/A'


def delete_id_data(index, name):
    try:
        ES.delete(index=index, id=name)
    except exceptions.ConnectionError as err:
        logger.exception(f"Connection Error : {err.message}")
    except exceptions.NotFoundError as err:
        logger.exception(f"NotFound Error : {err.message}")
    except Exception:
        logger.exception(f"Failed to delete id: {name} for index: {index}")
