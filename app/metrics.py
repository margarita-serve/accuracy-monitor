from utils import search_index
import time
import logging
import os
from datetime import timedelta
from dateutil import parser
from collections import Counter
import numpy as np
import pandas as pd
from sklearn import metrics
from utils import upsert_doc, search_data, LoadData, get_aggs, search_all_data, get_avg_aggs, get_count_inference, \
    make_percent

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def update_metrics(update_list, model_type, inference_name, positive_class=False, negative_class=False,
                   binary_threshold=None):
    for update_date in update_list:
        split_list = update_date.split(",")
        model_id = split_list[0]
        update_date = split_list[1]
        hour_added = timedelta(hours=1)
        end_time = parser.parse(update_date) + hour_added
        end_time = end_time.strftime('%Y-%m-%dT%H:00:00')
        query = {
            "range": {
                "timestamp": {
                    "gte": f"{update_date}",
                    "lt": f"{end_time}"
                }
            }
        }
        result, res = search_index(index=f"accuracy_match_data_{inference_name}_{model_id}", query=query)
        if result is False:
            return result, res
        actual = []
        predict = []
        for value in res:
            actual.append(value['_source']['actual_value'])
            predict.append(value['_source']['predict_value'])
        if len(actual) == 0 or len(predict) == 0:
            retry_count = 0
            while retry_count != 5 and len(actual) == 0:
                result, res = search_index(index=f"accuracy_match_data_{inference_name}_{model_id}", query=query)
                if result is True:
                    for value in res:
                        actual.append(value['_source']['actual_value'])
                        predict.append(value['_source']['predict_value'])
                else:
                    retry_count += 1
                    logger.info(f"Waiting for Elasticsearch Index update {retry_count}/5")
                    time.sleep(10)
            if len(actual) == 0 or len(predict) == 0:
                logger.warning(f"{update_date} value is empty. try again")

        if model_type == 'Regression':
            result, metrics_value = regression_metrics(actual, predict)
            if result is False:
                return result, metrics_value
        elif model_type == 'Binary':
            result, metrics_value = binary_metrics(actual, predict, 1, 0,
                                                   binary_threshold)
            if result is False:
                return result, metrics_value
        else:
            result, metrics_value = multiclass_metrics(actual, predict)
            if result is False:
                return result, metrics_value

        metrics_value['date'] = update_date
        result, message = upsert_doc(index=f"accuracy_aggregation_{inference_name}_{model_id}",
                                     name=update_date,
                                     doc=metrics_value)
        if result is False:
            return result, message

    return True, 'Success'


def regression_metrics(actual, predict):
    # 후처리하지않은 집계용 데이터 입니다.
    actual, predict = check_target(actual, predict)
    if actual is False:
        return False, predict
    count = len(actual)
    average_predicted = np.sum(predict)
    # 최종값 = average_predicted/count

    average_actual = np.sum(actual)
    # 최종값 = average_actual/count

    rmse = RMSE(actual, predict)
    # 최종값 = np.sqrt(rmse/count)
    # 값이 낮을수록 좋다.

    rmsle = RMSLE(actual, predict)
    # 최종값 = np.sqrt(rmsle/count)
    # 값이 낮을수록 좋다.

    mtd = mean_tweedie_deviance(actual, predict)
    # 최종값 = mtd/count
    # 양수값 0에 가까울수록 좋다

    mgd = mean_gamma_deviance(actual, predict)
    # 최종값 = mgd/count
    # 양수값 0에 가까울수록 좋다

    mae = MAE(actual, predict)
    # 최종값 = mae/count
    # 양수값 0에 가까울수록 좋다

    mape = MAPE(actual, predict)
    # 최종값 = mape/count
    # 퍼센트값 0에 가까울수록 좋다

    # r2_numerator, r2_denominator = r2(actual, predict)
    # 최종값 = 1 - ( r2_numerator/r2_denominator)
    # 음수값이 나올수도 있음 최고값은 1

    # d2_tweedie_numerator, d2_tweedie_denominator = d2_tweedie(actual, predict)
    # 최종값 = 1 - ( d2_tweedie_numerator/d2_twwedie_denominator)
    # 음수값이 나올수도 있음 최고값은 1

    # d2_absolute_error_numerator, d2_absolute_error_denominator = d2_absolute_error(actual, predict)
    # 최종값 = 1 - ( d2_absolute_error_numerator/d2_absolute_error_denominator)
    # 음수값이 나올수도 있음 최고값은 1

    metric_value = {
        "average_predicted": average_predicted,
        "average_actual": average_actual,
        "count": count,
        "rmse": rmse,
        "rmsle": rmsle,
        "mae": mae,
        "mape": mape,
        "mean_tweedie_deviance": mtd,
        "mean_gamma_deviance": mgd
    }
    return True, metric_value


def binary_metrics(actual, predict, positive_class, negative_class, binary_threshold):
    # 후처리하지않은 집계용 데이터 입니다.
    if type(actual[0]) is int:
        positive_class = int(positive_class)
        negative_class = int(negative_class)
        np.array(predict, dtype=int)
    elif type(actual[0]) is float:
        positive_class = float(positive_class)
        negative_class = float(negative_class)
        predict = np.array(predict, dtype=float)

    percentage_predicted = Counter(predict)[positive_class]
    percentage_actual = Counter(actual)[positive_class]

    actual, predict = check_target(actual, predict)
    if actual is False:
        return False, predict

    tn, fp, fn, tp = metrics.confusion_matrix(actual, predict, labels=[negative_class, positive_class]).ravel()
    count = len(actual)

    metric_value = {
        "percentage_predicted": percentage_predicted,
        "percentage_actual": percentage_actual,
        "count": count,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }
    return True, metric_value


def multiclass_metrics(actual, predict):
    return True, 'value'


def check_target(actual, predict):
    actual = np.array(actual)
    predict = np.array(predict)
    if actual.ndim == 1:
        actual = actual.reshape((-1, 1))
    if predict.ndim == 1:
        predict = predict.reshape((-1, 1))

    if actual.shape[1] != predict.shape[1]:
        return False, "y_true and y_pred have different number of output ({0}!={1})".format(
            actual.shape[1], predict.shape[1]
        )
    return actual, predict


def RMSE(actual, predict):
    rmse = np.sum((actual - predict) ** 2)
    return rmse


def RMSLE(actual, predict):
    if (actual < 0).any() or (predict < 0).any():
        return "N/A"
    rmsle = np.sum((np.log1p(actual) - np.log1p(predict)) ** 2)
    return rmsle


def MAE(actual, predict):
    mae = np.sum(np.abs(actual - predict))
    return mae


def MAPE(actual, predict):
    epsilon = np.finfo(np.float64).eps
    mape = np.sum(np.abs(actual - predict) / np.maximum(np.abs(actual), epsilon))
    return mape


def mean_tweedie_deviance(actual, predict):
    if (actual < 0).any() or (predict < 0).any():
        return "N/A"
    mtd = np.sum((actual - predict) ** 2)
    return mtd


def mean_gamma_deviance(actual, predict):
    if (actual < 0).any() or (predict < 0).any():
        return "N/A"
    mgd = np.sum(-np.log(actual / predict) + (actual - predict) / predict) * 2
    return mgd


def r2(actual, predict):
    numerator = ((actual - predict) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((actual - np.average(actual, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
    return numerator[0], denominator[0]


def d2_tweedie(actual, predict):
    actual = np.squeeze(actual)
    predict = np.squeeze(predict)
    numerator = metrics.mean_tweedie_deviance(actual, predict)
    actual_avg = np.average(actual)
    denominator = np.average((actual - actual_avg) ** 2)
    return numerator, denominator


def d2_absolute_error(actual, predict):
    numerator = metrics.mean_pinball_loss(
        actual,
        predict,
        alpha=0.5,
        multioutput="raw_values"
    )
    actual_quantile = np.tile(
        np.percentile(actual, q=0.5 * 100, axis=0), (len(actual), 1)
    )
    denominator = metrics.mean_pinball_loss(
        actual,
        actual_quantile,
        alpha=0.5,
        multioutput="raw_values"
    )
    return numerator[0], denominator[0]


def get_accuracy_metrics(inference_name, model_id, start_time, end_time, model_type):
    aggs = get_aggs(model_type)
    result, value = search_data(f"accuracy_aggregation_{inference_name}_{model_id}", start_time, end_time, aggs)
    if result is False:
        return False, value

    result, base_metrics = LoadData('accuracy_base_metric', f"{inference_name}_{model_id}").load_data()
    if result is False:
        return False, base_metrics

    merge_value = {}
    if model_type == 'Regression':
        count = value.pop('count')
        count = count['value']

        for v in value:
            if v == 'rmse':
                merge_value[v] = {}
                merge_value[v]['base'] = base_metrics[v]
                merge_value[v]['actual'] = np.sqrt(value[v]['value'] / count)
            elif v == 'rmsle':
                merge_value[v] = {}
                merge_value[v]['base'] = base_metrics[v]
                try:
                    merge_value[v]['actual'] = np.sqrt(value[v]['value'] / count)
                except:
                    merge_value[v]['actual'] = "N/A"
            elif v == 'mean_tweedie_deviance' or v == 'mean_gamma_deviance':
                merge_value[v] = {}
                merge_value[v]['base'] = base_metrics[v]
                try:
                    merge_value[v]['actual'] = value[v]['value'] / count
                except:
                    merge_value[v]['actual'] = "N/A"
            else:
                merge_value[v] = {}
                merge_value[v]['base'] = base_metrics[v]
                merge_value[v]['actual'] = value[v]['value'] / count
            per_value = make_percent(base_metrics, v, merge_value[v]['actual'])
            merge_value[v]['percent'] = per_value

    elif model_type == 'Binary':
        tn = value['tn']['value']
        fn = value['fn']['value']
        fp = value['fp']['value']
        tp = value['tp']['value']

        tpr = tp / (tp + fn)
        tpr_per = make_percent(base_metrics, 'tpr', tpr)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy_per = make_percent(base_metrics, 'accuracy', accuracy)
        f1 = 2 * tp / (2 * tp + fp + fn)
        f1_per = make_percent(base_metrics, 'f1', f1)
        ppv = tp / (tp + fp)
        ppv_per = make_percent(base_metrics, 'ppv', ppv)
        fnr = fn / (fn + tp)
        fnr_per = make_percent(base_metrics, 'fnr', fnr)
        fpr = fp / (fp + tn)
        fpr_per = make_percent(base_metrics, 'fpr', fpr)

        merge_value = {
            'tpr': {
                'base': base_metrics['tpr'],
                'actual': tpr,
                'percent': tpr_per
            },
            'accuracy': {
                'base': base_metrics['accuracy'],
                'actual': accuracy,
                'percent': accuracy_per
            },
            'f1': {
                'base': base_metrics['f1'],
                'actual': f1,
                'percent': f1_per
            },
            'ppv': {
                'base': base_metrics['ppv'],
                'actual': ppv,
                'percent': ppv_per
            },
            'fnr': {
                'base': base_metrics['fnr'],
                'actual': fnr,
                'percent': fnr_per
            },
            'fpr': {
                'base': base_metrics['fpr'],
                'actual': fpr,
                'percent': fpr_per
            },
        }

    else:
        pass

    return True, merge_value


def get_accuracy_timeline(inference_name, model_id, start_time, end_time, model_type):
    start_time, end_time, interval, offset = get_interval(start_time, end_time)
    aggs = get_aggs(model_type)

    date_aggs = {
        "timeline": {
            "date_histogram": {
                "field": "date",
                "fixed_interval": interval,
                "extended_bounds": {
                    "min": start_time,
                    "max": end_time
                },
                "offset": offset
            },
            "aggs": aggs
        }
    }

    result, value = search_data(f"accuracy_aggregation_{inference_name}_{model_id}", start_time, end_time, date_aggs)
    if result is False:
        return False, value

    result, base_metrics = LoadData('accuracy_base_metric', f"{inference_name}_{model_id}").load_data()
    if result is False:
        return False, base_metrics

    if model_type == 'Regression':
        actual, date_list = make_regression_actual(value)
    elif model_type == 'Binary':
        actual, date_list = make_binary_actual(value)
    else:
        actual = None
        date_list = None

    total_value = {"base": base_metrics, "actual": actual, "date": date_list}

    return True, total_value


def get_interval(start_time, end_time):
    start_parse = parser.parse(start_time)
    start_time = start_parse.strftime('%Y-%m-%dT%H:00:00.000Z')
    start_hour = start_parse.hour
    end_parse = parser.parse(end_time)
    end_time = end_parse.strftime('%Y-%m-%dT%H:00:00.000Z')

    term = end_parse - start_parse
    term_day = term.days
    if term_day < 7:
        return start_time, end_time, '1h', "+0h"
    elif 7 <= term_day < 60:
        return start_time, end_time, '1d', f"+{start_hour}h"
    elif 60 <= term_day < 730:
        return start_time, end_time, '7d', f"+{start_hour}h"
    else:
        return start_time, end_time, '30d', f"+{start_hour}h"


def make_date_list(start_time, end_time, interval):
    # Todo interval이 day 이상일경우 시간 처리
    if interval == '1h':
        freq = 'h'
        form = '%Y-%m-%dT%H:00:00.000Z'
    elif interval == '1d':
        freq = 'd'
        form = '%Y-%m-%dT%H:00:00.000Z'
    elif interval == '7d':
        freq = '7d'
        form = '%Y-%m-%dT%H:00:00.000Z'
    else:
        freq = '30d'
        form = '%Y-%m-%dT%H:00:00.000Z'
    date_list = pd.date_range(start_time, end_time, freq=freq).strftime(form).tolist()

    return date_list


def make_regression_actual(value):
    actual = []
    date_list = []

    for v in value['timeline']['buckets']:
        date_list.append(v['key_as_string'])
        if v['count']['value'] == 0:
            actual.append(
                {
                    "average_predicted": 0,
                    "average_actual": 0,
                    "count": 0,
                    "rmse": 'N/A',
                    "rmsle": 'N/A',
                    "mae": 'N/A',
                    # "median_absolute_error": 'N/A',
                    "mape": 'N/A',
                    "mean_tweedie_deviance": 'N/A',
                    "mean_gamma_deviance": 'N/A'
                }
            )
        else:
            try:
                rmsle = np.sqrt(v['rmsle']['value'] / v['count']['value'])
                mtd = v['mean_tweedie_deviance']['value'] / v['count']['value']
                mgd = v['mean_gamma_deviance']['value'] / v['count']['value']
            except:
                rmsle = "N/A"
                mtd = "N/A"
                mgd = "N/A"
            actual.append(
                {
                    "average_predicted": v['average_predicted']['value'] / v['count']['value'],
                    "average_actual": v['average_actual']['value'] / v['count']['value'],
                    "count": v['count']['value'],
                    "rmse": np.sqrt(v['rmse']['value'] / v['count']['value']),
                    "rmsle": rmsle,
                    "mae": v['mae']['value'] / v['count']['value'],
                    # "median_absolute_error": v['median_absolute_error']['value'] / v['count']['value'],
                    "mape": v['mape']['value'] / v['count']['value'],
                    "mean_tweedie_deviance": mtd,
                    "mean_gamma_deviance": mgd
                }
            )

    return actual, date_list


def make_binary_actual(value):
    actual = []
    date_list = []

    for v in value['timeline']['buckets']:
        date_list.append(v['key_as_string'])
        tp = v['tp']['value']
        fn = v['fn']['value']
        fp = v['fp']['value']
        tn = v['tn']['value']
        percentage_predicted_count = v['percentage_predicted_count']['value']
        percentage_actual_count = v['percentage_actual_count']['value']

        if v['count']['value'] == 0:
            actual.append(
                {
                    "count": 0,
                    "percentage_predicted": 0,
                    "percentage_actual": 0,
                    "tpr": 'N/A',
                    "accuracy": 'N/A',
                    "f1": 'N/A',
                    "ppv": 'N/A',
                    "fnr": 'N/A',
                    "fpr": 'N/A'
                }
            )
        else:
            actual.append(
                {
                    "count": v['count']['value'],
                    "percentage_predicted": round(percentage_predicted_count / v['count']['value'] * 100, 3),
                    "percentage_actual": round(percentage_actual_count / v['count']['value'] * 100, 3),
                    "tpr": tp / (tp + fn),
                    "accuracy": (tp + tn) / (tp + tn + fp + fn),
                    "f1": 2 * tp / (2 * tp + fp + fn),
                    "ppv": tp / (tp + fp),
                    "fnr": fn / (fn + tp),
                    "fpr": fp / (fp + tn)
                }
            )

    return actual, date_list


def get_accuracy_monitor(inference_name, model_id, drift_metric, model_type):
    if model_type == "Regression":
        aggs = {
            "count": {
                "sum": {
                    "field": "count"
                }
            },
            drift_metric: {
                "sum": {
                    "field": drift_metric
                }
            }
        }
    elif model_type == 'Binary':
        aggs = {
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
        aggs = {
            "count": {
                "sum": {
                    "field": "count"
                }
            }
        }

    result, value = search_all_data(f"accuracy_aggregation_{inference_name}_{model_id}", aggs)
    if result is False:
        return False, 0

    if value['count']['value'] == 0:
        return False, 0

    if model_type == "Regression":
        try:
            if drift_metric == 'rmse' or drift_metric == 'rmsle':
                metric_value = np.sqrt(value[drift_metric]['value'] / value['count']['value'])
            else:
                metric_value = value[drift_metric]['value'] / value['count']['value']
        except:
            metric_value = "N/A"

    elif model_type == "Binary":
        tp = value['tp']['value']
        tn = value['tn']['value']
        fp = value['fp']['value']
        fn = value['fn']['value']

        if drift_metric == "tpr":
            metric_value = tp / (tp + fn)
        elif drift_metric == "accuracy":
            metric_value = (tp + tn) / (tp + tn + fp + fn)
        elif drift_metric == "f1":
            metric_value = 2 * tp / (2 * tp + fp + fn)
        elif drift_metric == "ppv":
            metric_value = tp / (tp + fp)
        elif drift_metric == "fnr":
            metric_value = fn / (fn + tp)
        elif drift_metric == "fpr":
            metric_value = fp / (fp + tn)
        else:
            metric_value = "N/A"

    else:
        # todo Multiclass
        metric_value = 0

    return metric_value, value['count']['value']


def get_predicted_actual(inference_name, model_id, model_type, start_time, end_time):
    start_time, end_time, interval, offset = get_interval(start_time, end_time)

    aggs = get_avg_aggs(model_type)
    date_aggs = {
        "timeline": {
            "date_histogram": {
                "field": "date",
                "fixed_interval": interval,
                "extended_bounds": {
                    "min": start_time,
                    "max": end_time
                },
                "offset": offset
            },
            "aggs": aggs
        }
    }

    result, aggregation_data = search_data(f"accuracy_aggregation_{inference_name}_{model_id}", start_time,
                                           end_time, date_aggs)
    if result is False:
        return False, aggregation_data

    date_aggs = {
        "timeline": {
            "date_histogram": {
                "field": "timestamp",
                "fixed_interval": interval,
                "extended_bounds": {
                    "min": start_time,
                    "max": end_time
                },
                "offset": offset
            }
        }
    }

    result, total_count = get_count_inference(f"accuracy_inference_data_{inference_name}", start_time, end_time,
                                              date_aggs)
    if result is False:
        return False, total_count

    value = []
    date_list = []
    for v in aggregation_data['timeline']['buckets']:
        date_list.append(v['key_as_string'])
        if model_type == "Regression":
            value.append({
                "count": v['count']['value'],
                "average_predicted": v['average_predicted']['value'] / v['count']['value'] if v['count'][
                                                                                                  'value'] != 0 else 0,
                "average_actual": v['average_actual']['value'] / v['count']['value'] if v['count']['value'] != 0 else 0,
            })
        elif model_type == "Binary":
            value.append({
                "count": v['count']['value'],
                "percentage_predicted": v['percentage_predicted']['value'] / v['count']['value'] if v['count'][
                                                                                                        'value'] != 0 else 0,
                "percentage_actual": v['percentage_actual']['value'] / v['count']['value'] if v['count'][
                                                                                                  'value'] != 0 else 0,
            })
        else:
            pass

    count = []
    for v in total_count['timeline']['buckets']:
        count.append({
            "total_count": v['doc_count']
        })

    for i in range(len(value)):
        value[i] = {**value[i], **count[i]}

    result, base_metrics = LoadData('accuracy_base_metric', f"{inference_name}_{model_id}").load_data()
    if result is False:
        return False, base_metrics

    data = {
        "data": value,
        "date": date_list,
        "base": base_metrics
    }
    return True, data
