import numpy as np
import utils
import logging
import os
from sklearn import metrics
from collections import Counter

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def create_info(inference_name, model_id, target, association_id, association_id_in_feature, df, model_type,
                positive_class=False, negative_class=False, binary_threshold=False):
    df_list = list(df.columns)
    if association_id_in_feature is True:
        if association_id not in df_list:
            logger.critical(f"Association ID: {association_id} not in columns")
            return False, f"Association ID: {association_id} not in columns"
        association_index = df_list.index(association_id)
    else:
        association_index = None

    if model_type == 'Regression':
        info = {"association_id": association_id, "target": target, "association_index": association_index,
                'model_type': model_type, "columns": df_list}
    elif model_type == 'Binary':
        info = {"association_id": association_id, "target": target, "association_index": association_index,
                'model_type': model_type, "positive_class": positive_class,
                'negative_class': negative_class, 'binary_threshold': binary_threshold, "columns": df_list}
    else:
        info = {"association_id": association_id, "target": target, "association_index": association_index,
                'model_type': model_type, "columns": df_list}
    result, message = utils.save_data('accuracy_monitor_info', f"{inference_name}_{model_id}", info)
    if result is False:
        return False, message

    return True, ""


def create_monitor_setting(inference_name, model_id, drift_metric, drift_measurement, atrisk_value, failing_value):
    setting = {
        "current_model": model_id,
        "drift_metric": drift_metric,
        "drift_measurement": drift_measurement,
        "atrisk_value": atrisk_value,
        "failing_value": failing_value,
        "status": "enable"
    }
    result, message = utils.save_data('accuracy_monitor_setting', inference_name, setting)
    if result is False:
        return False, message
    else:
        return True, message


def create_base(model, df, target_df, inference_name, model_id, model_type, framework,
                positive_class=False, negative_class=False, binary_threshold=False):
    pred = utils.get_prediction(df, model, framework)
    if pred is False:
        logger.warning("Model prediction Error")
        return False
    result, metric_value = make_metrics(pred, target_df, model_type, positive_class, negative_class, binary_threshold)
    if result is False:
        return False, metric_value
    result, message = utils.save_data('accuracy_base_metric', f"{inference_name}_{model_id}", metric_value)
    if result is False:
        return False, message

    utils.make_index(f"accuracy_aggregation_{inference_name}_{model_id}")

    return True, "Success"


def make_metrics(pred, target_df, model_type, positive_class=False, negative_class=False, binary_threshold=None):
    truth = target_df.to_list()
    length = len(pred)
    if model_type == 'Regression':
        try:
            rmsle = metrics.mean_squared_log_error(truth, pred, squared=False)
            mean_tweedie_deviance = metrics.mean_tweedie_deviance(truth, pred)
            mean_gamma_deviance = metrics.mean_gamma_deviance(truth, pred)
        except:
            rmsle = "N/A"
            mean_tweedie_deviance = "N/A"
            mean_gamma_deviance = "N/A"

        metric_value = {
            "average_predicted": np.mean(pred),
            "average_actual": np.mean(truth),
            "count": length,
            "rmse": metrics.mean_squared_error(truth, pred, squared=False),
            "rmsle": rmsle,
            "mae": metrics.mean_absolute_error(truth, pred),
            # "median_absolute_error": metrics.median_absolute_error(truth, pred),
            "mape": metrics.mean_absolute_percentage_error(truth, pred),
            "mean_tweedie_deviance": mean_tweedie_deviance,
            "mean_gamma_deviance": mean_gamma_deviance,
        }
    elif model_type == 'Binary':
        # try:
        #     if type(truth[0]) is int:
        #         positive_class = int(positive_class)
        #         negative_class = int(negative_class)
        #     elif type(truth[0]) is float:
        #         positive_class = float(positive_class)
        #         negative_class = float(negative_class)
        # except Exception as err:
        #     return False, f"truth data type: {type(truth[0])} but positive class type: {type(positive_class)}"

        if binary_threshold:
            for i in range(len(pred)):
                if pred[i] >= binary_threshold:
                    pred[i] = 1
                else:
                    pred[i] = 0

            tn, fp, fn, tp = metrics.confusion_matrix(truth, pred, labels=[0, 1]).ravel()

        else:
            tn, fp, fn, tp = metrics.confusion_matrix(truth, pred, labels=[0, 1]).ravel()

        metric_value = {
            "count": length,
            "percentage_predicted": round(Counter(pred)[1] / length * 100, 3),
            "percentage_actual": round(Counter(truth)[1] / length * 100, 3),
            "tpr": metrics.recall_score(truth, pred, pos_label=1),
            "accuracy": metrics.accuracy_score(truth, pred),
            "f1": metrics.f1_score(truth, pred, pos_label=1),
            "ppv": metrics.precision_score(truth, pred, pos_label=1),
            "fnr": 1 - metrics.recall_score(truth, pred, pos_label=1),
            "fpr": fp / (fp + tn)
        }

    else:
        # Multiclass
        metric_value = {
            "average_predicted": np.mean(pred),
            "average_actual": np.mean(truth),
            "count": length,
        }

    return True, metric_value
