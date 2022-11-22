import logging
import os
import time
from multiprocessing import Process

from flask import Flask
from flask_cors import CORS
from flask_restx import Api, Resource, fields, reqparse

import utils
from metrics import update_metrics, get_accuracy_metrics, get_accuracy_timeline, get_accuracy_monitor, \
    get_predicted_actual
from kafka_func import consumeKafka, makeproducer, produceKafka
from create_info import create_info, create_monitor_setting, create_base
from upload_actual import upload_data

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()

app = Flask(__name__)
CORS(app)
api = Api(app, version='1.0', title='KoreServe Accuracy Monitor Service API')

ns = api.namespace('', describtion='REST-API operations')

#######################################################################
# restX Input Model
#######################################################################

init_data_model = api.model('InitData', {
    "dataset_path": fields.String(example='dataset/train_data.csv', required=True),
    "model_path": fields.String(example='testmodel/mpg2/1', required=True),
    "inference_name": fields.String(example="mpg-sample", required=True),
    "model_id": fields.String(example="000001", required=True),
    "target_label": fields.String(example="MPG", required=True),
    "association_id": fields.String(example='index', required=True),
    "association_id_in_feature": fields.Boolean(example="true", required=True),
    "model_type": fields.String(enum=['Regression', 'Multiclass', 'Binary'], example="Regression", required=True),
    "framework": fields.String(enum=['TensorFlow', 'SkLearn', 'XGBoost', 'LightGBM', 'PyTorch'],
                               example="TensorFlow", required=True),
    "drift_metric": fields.String(
        enum=['rmse', 'rmsle', 'mae', 'mape', 'mean_tweedie_deviance', 'gamma_deviance', 'tpr', 'accuracy',
              'f1', 'ppv', 'fnr', 'fpr'],
        example='rmse', required=True),
    "drift_measurement": fields.String(enum=['value', 'percent'], example='value', required=True),
    "atrisk_value": fields.Float(example=10, required=True),
    "failing_value": fields.Float(example=15, required=True),
    "positive_class": fields.String(example="1"),
    "negative_class": fields.String(example="0"),
    "binary_threshold": fields.Float(example=0.5)
})

ground_data_model = api.model('GroundData', {
    "inference_name": fields.String(example='mpg-sample', required=True),
    "dataset_path": fields.String(example='grounddata/dataset.csv', required=True),
    "actual_response": fields.String(example='MPG', required=True),
    "association_column": fields.String(example='index', required=True)
})

monitor_setting_model = api.model('MonitorSetting', {
    "current_model": fields.String(example='000001'),
    "drift_metric": fields.String(
        enum=['rmse', 'rmsle', 'mae', 'mape', 'mean_tweedie_deviance', 'mean_gamma_deviance', 'tpr', 'accuracy',
              'f1', 'ppv', 'fnr', 'fpr'],
        example='rmse', required=True),
    "drift_measurement": fields.String(enum=['value', 'percent'], example='value', required=True),
    "atrisk_value": fields.Float(example=10, required=True),
    "failing_value": fields.Float(example=15, required=True)
})

update_association_model = api.model("AssociationID", {
    "association_id": fields.String(example="index")
})

#######################################################################
# restX Output Model
#######################################################################
base_output_model = api.model("BaseOutputModel", {
    "message": fields.String,
    "inference_name": fields.String,
})

accuracy_metrics_output_model = api.model("AccuracyMetricsOutputModel", {
    "message": fields.String,
    "data": fields.String,
    "start_time": fields.String,
    "end_time": fields.String
})

info_data_output_model = api.model("InfoDataOutputModel", {
    "message": fields.String,
    "data": fields.String,
    "inference_name": fields.String
})


############################################################
# HTTP Routing
############################################################


@ns.route("/accuracy-monitor/accuracy/<string:inferencename>")
@ns.param('end_time', 'example=2022-05-13:05', required=True)
@ns.param('start_time', 'example=2022-05-04:05', required=True)
@ns.param('type', 'timeline or aggregation', required=True)
@ns.param('model_history_id', 'Model History ID', required=True)
@ns.param('inferencename', 'Kserve Inferencename')
class AccuracyMonitorSearch(Resource):
    @ns.marshal_with(accuracy_metrics_output_model, code=200, skip_none=True)
    def get(self, inferencename):
        parser = reqparse.RequestParser()
        parser.add_argument('start_time', required=True, type=str,
                            location='args', help='2022-05-04:05')
        parser.add_argument('end_time', required=True, type=str,
                            location='args', help='2022-05-13:05')
        parser.add_argument('type', required=True, type=str,
                            location='args', help='timeline or aggregation')
        parser.add_argument('model_history_id', required=True, type=str,
                            location='args', help='000001')

        args = parser.parse_args()

        start_time, end_time, value_type, model_history_id = parsing_accuracy_data(args)
        try:
            start_time = utils.convertTimestamp(start_time)
            end_time = utils.convertTimestamp(end_time)
        except:
            return {"message": "time parser error, time format must be yyyy-mm-dd:hh"}, 400

        result, inference_info = utils.LoadData('accuracy_monitor_info',
                                                f"{inferencename}_{model_history_id}").load_data()
        if result is False:
            return {"message": "Get Monitor Info Failed", "data": inference_info, "inference_name": inferencename}, 400

        model_type = inference_info['model_type']

        utils.make_index(f"accuracy_aggregation_{inferencename}_{model_history_id}")  # 수정본 임시 추가 추후 삭제

        if value_type == 'aggregation':
            result, value = get_accuracy_metrics(inferencename, model_history_id, start_time, end_time, model_type)
        else:
            result, value = get_accuracy_timeline(inferencename, model_history_id, start_time, end_time, model_type)

        if result is False:
            if str(value).find("NotFoundError") >= 0:
                return {"message": value, "inference_name": inferencename}, 404
            elif str(value).find("No data") >= 0:
                return {"message": value, "inference_name": inferencename, "start_time": start_time,
                        "end_time": end_time, "data": []}, 200
            return {"message": value, "inference_name": inferencename}, 400

        return {"message": "Success get accuracy metrics", "inference_name": inferencename,
                "start_time": start_time, "end_time": end_time, "data": value}, 200


@ns.route("/accuracy-monitor/predicted-actual/<string:inferencename>")
@ns.param('end_time', 'example=2022-05-13:05', required=True)
@ns.param('start_time', 'example=2022-05-04:05', required=True)
@ns.param('model_history_id', 'Model History ID', required=True)
@ns.param('inferencename', 'Kserve Inferencename')
class PredictedActual(Resource):
    @ns.marshal_with(accuracy_metrics_output_model, code=200, skip_none=True)
    def get(self, inferencename):
        parser = reqparse.RequestParser()
        parser.add_argument('start_time', required=True, type=str,
                            location='args', help='2022-05-04:01')
        parser.add_argument('end_time', required=True, type=str,
                            location='args', help='2022-05-13:01')
        parser.add_argument('model_history_id', required=True, type=str,
                            location='args', help='000001')

        args = parser.parse_args()

        start_time, end_time, _, model_history_id = parsing_accuracy_data(args)
        try:
            start_time = utils.convertTimestamp(start_time)
            end_time = utils.convertTimestamp(end_time)
        except:
            return {"message": "time parser error, time format must be yyyy-mm-dd:hh"}, 400

        result, inference_info = utils.LoadData('accuracy_monitor_info',
                                                f"{inferencename}_{model_history_id}").load_data()
        if result is False:
            return {"message": "Get Monitor Info Failed", "data": inference_info, "inference_name": inferencename}, 400

        model_type = inference_info['model_type']

        result, data = get_predicted_actual(inferencename, model_history_id, model_type, start_time, end_time)
        if result is False:
            if str(data).find("NotFoundError") >= 0:
                return {"message": data, "inference_name": inferencename}, 404
            elif str(data).find("No data") >= 0:
                return {"message": data, "inference_name": inferencename, "start_time": start_time,
                        "end_time": end_time, "data": []}, 200
            return {"message": data, "inference_name": inferencename}, 400

        return {"message": "Success get predicted actual data", "data": data, "inference_name": inferencename}, 200


@ns.route("/accuracy-monitor/monitor-info/<string:inferencename>")
@ns.param('model_history_id', 'Model History ID', required=True)
@ns.param('inferencename', 'Kserve Inferencename')
class GetMonitorInfo(Resource):
    @ns.marshal_with(accuracy_metrics_output_model, code=200, skip_none=True)
    def get(self, inferencename):
        parser = reqparse.RequestParser()
        parser.add_argument('model_history_id', required=True, type=str,
                            location='args', help='000001')
        args = parser.parse_args()
        model_history_id = args.get('model_history_id')
        result, inference_info = utils.LoadData('accuracy_monitor_info',
                                                f"{inferencename}_{model_history_id}").load_data()

        if result is False:
            return {"message": "Get Monitor Info Failed", "data": inference_info, "inference_name": inferencename}, 400

        return {"message": "Get Monitor Info Success", "data": inference_info, "inference_name": inferencename}, 200


@ns.route("/accuracy-monitor")
class AccuracyMonitorInit(Resource):
    @ns.expect(init_data_model, validate=True)
    @ns.marshal_with(base_output_model, code=201, skip_none=True)
    def post(self):
        args = api.payload
        dataset_path, model_path, inference_name, model_id, target_label, association_id, association_id_in_feature, \
        model_type, framework, drift_metric, drift_measurement, atrisk_value, failing_value = parsing_base_data(args)

        logger.info('-----------------------------------------------')
        logger.info(inference_name, drift_metric)
        logger.info('-----------------------------------------------')

        positive_class = args.get('positive_class')
        negative_class = args.get('negative_class')
        binary_threshold = args.get('binary_threshold')

        try:
            df = utils.read_csv(dataset_path)
            if df is False:
                return {"message": "Dataset Load Error", "inference_name": inference_name}, 400

            model = utils.load_model(framework, model_path)
            if model is False:
                return {"message": "Model Load Error", "inference_name": inference_name}, 400

            result, message = utils.validate(model, df, target_label, framework)
            if result is False:
                return {"message": message, "inference_name": inference_name}, 400

            df, target_df = split_df(df, target_label)
            status, message = create_info(inference_name, model_id, target_label, association_id,
                                          association_id_in_feature, df, model_type,
                                          positive_class, negative_class, binary_threshold)
            if status is False:
                return {"message": message, "inference_name": inference_name}, 400

            result, message = create_base(model, df, target_df, inference_name, model_id, model_type, framework,
                                          positive_class, negative_class, binary_threshold)
            if result is False:
                utils.delete_id_data("accuracy_monitor_info", f"{inference_name}_{model_id}")
                return {"message": message, "inference_name": inference_name}, 400

            result, message = create_monitor_setting(inference_name, model_id, drift_metric, drift_measurement,
                                                     atrisk_value, failing_value)
            if result is False:
                utils.delete_id_data("accuracy_monitor_info", f"{inference_name}_{model_id}")
                utils.delete_id_data("accuracy_base_metric", f"{inference_name}_{model_id}")
                return {"message": message, "inference_name": inference_name}, 400

        except Exception as err:
            utils.delete_id_data("accuracy_monitor_info", f"{inference_name}_{model_id}")
            utils.delete_id_data("accuracy_base_metric", f"{inference_name}_{model_id}")
            return {"message": err, "inference_name": inference_name}, 500

        logger.info(f" Accuracy Monitor Inference base data init Success, {inference_name}")
        return {"message": 'Accuracy Monitor Inference base data init Success',
                "inference_name": inference_name}, 201


@ns.route("/accuracy-monitor/actual")
class UploadTruth(Resource):
    @ns.expect(ground_data_model, validate=True)
    @ns.marshal_with(base_output_model, code=201, skip_none=True)
    def post(self):
        args = api.payload

        inference_name = args.get('inference_name')
        path = args.get('dataset_path')
        actual_response = args.get("actual_response")
        association_column = args.get('association_column')

        local_dataset_path = utils.minio_client(path)
        _, monitor_setting = utils.LoadData('accuracy_monitor_setting', inference_name).load_data()
        current_model = monitor_setting['current_model']
        _, inference_info = utils.LoadData('accuracy_monitor_info', f"{inference_name}_{current_model}").load_data()
        association_id = inference_info['association_id']
        association_index = inference_info['association_index']
        model_type = inference_info['model_type']
        positive_class = False
        negative_class = False
        binary_threshold = None

        # if association_column == "" or association_column is None:
        #     association_column = "associationID"

        if model_type == 'Binary':
            positive_class = inference_info['positive_class']
            negative_class = inference_info['negative_class']
            binary_threshold = inference_info['binary_threshold']

        df = utils.read_csv(local_dataset_path)

        result, message = utils.validate_actual(df, actual_response, association_column, positive_class, negative_class)
        if result is False:
            return {"message": message, "inference_name": inference_name}, 400

        result, message = utils.check_index("accuracy_inference_data", inference_name)
        if result is False:
            return {"message": "No prediction data", "inference_name": inference_name}, 400

        if actual_response not in df.columns:
            return {"message": f'Actual Response: {actual_response} not in columns',
                    "inference_name": inference_name}, 400

        if association_column not in df.columns:
            return {"message": f'Association Column: {association_column} not in columns',
                    "inference_name": inference_name}, 400

        match_df = df[[association_column, actual_response]]

        result, message, update_list, count = upload_data(match_df, inference_name, association_id, association_column,
                                                          association_index, actual_response, positive_class,
                                                          negative_class, binary_threshold)
        if result is False:
            if message == 404:
                return {"message": "No prediction data found", "inferencename": inference_name}, 404
            return {"message": message, "inference_name": inference_name}, 400
        result, message = update_metrics(update_list, model_type, inference_name, positive_class, negative_class,
                                         binary_threshold)
        if result is False:
            return {"message": message, "inference_name": inference_name}, 400

        return {"message": f'Actual data matching was successful count:{count} ', "inference_name": inference_name}, 201


@ns.route("/accuracy-monitor/<string:inferencename>")
class PatchMonitorSetting(Resource):
    @ns.expect(monitor_setting_model, validate=True)
    @ns.marshal_with(base_output_model, code=200, skip_none=True)
    def patch(self, inferencename):
        args = api.payload
        if args.get('current_model') is not None:
            result, status = utils.check_index(inferencename, args.get('current_model'))
            if result is False:
                return status
            if status is False:
                return {"message": "Update failed. Register the model information to update first.",
                        "inferencename": inferencename}, 400

        result, message = utils.update_data('accuracy_monitor_setting', inferencename, args)
        if result is False:
            return {"message": message, "inference_name": inferencename}, 400
        else:
            return {"message": "Monitoring settings update was successful.", "inference_name": inferencename}, 200


# @ns.route("/accuracy-monitor/<string:inferencename>/association-id")
# class UpdateAssociationID(Resource):
#     @ns.expect(update_association_model, validate=True)
#     @ns.marshal_with(base_output_model, code=200, skip_none=True)
#     def patch(self, inferencename):
#         args = api.payload
#
#         association_id = args.get('association_id')
#         if association_id == "":
#             association_id = None
#         result = update_association(inferencename, association_id)
#         if result is False:
#             return {"message": "AssociationID changed Failed", "inference_name": inferencename}, 400
#         return {"message": "AssociationID chaged Success", "inference_name": inferencename}, 200


@ns.route("/accuracy-monitor/disable-monitor/<string:inferencename>")
@ns.param("inferencename", "Kserve Inferencename")
class DisableMonitor(Resource):
    @ns.marshal_with(base_output_model, code=200, skip_none=True)
    def patch(self, inferencename):
        result = disable_monitor(inferencename)
        if result is False:
            return {"message": "Disable Failed", "inference_name": inferencename}, 400

        return {"message": "Accuracy Monitor is disable", "inference_name": inferencename}, 200


@ns.route("/accuracy-monitor/enable-monitor/<string:inferencename>")
@ns.param("inferencename", "Kserve Inferencename")
class EnableMonitor(Resource):
    @ns.marshal_with(base_output_model, code=200, skip_none=True)
    def patch(self, inferencename):
        result = enable_monitor(inferencename)
        if result is False:
            return {"message": "Enable Failed", "inference_name": inferencename}, 400

        return {"message": "Accuracy Monitor is enabled", "inference_name": inferencename}, 200


############################################################
# Domain Logic
############################################################


def split_df(df, target):
    target_df = df.pop(target)
    return df, target_df


def parsing_base_data(args):
    model_path = args.get('model_path')
    dataset_path = args.get('dataset_path')
    inference_name = args.get('inference_name')
    model_id = args.get('model_id')
    target_label = args.get('target_label')
    association_id = args.get('association_id')
    association_id_in_feature = args.get('association_id_in_feature')
    model_type = args.get('model_type')
    framework = args.get('framework')
    drift_metric = args.get('drift_metric')
    drift_measurement = args.get('drift_measurement')
    atrisk_value = args.get('atrisk_value')
    failing_value = args.get('failing_value')

    local_model_path = utils.minio_client(model_path)
    local_dataset_path = utils.minio_client(dataset_path)

    return local_dataset_path, local_model_path, inference_name, model_id, target_label, association_id, association_id_in_feature, \
           model_type, framework, drift_metric, drift_measurement, atrisk_value, failing_value


def parsing_accuracy_data(request_args):
    start_time = request_args.get('start_time')
    end_time = request_args.get('end_time')
    value_type = request_args.get('type')
    model_history_id = request_args.get('model_history_id')

    return start_time, end_time, value_type, model_history_id


def get_monitor_settings(inference_name):
    monitor_setting_data = utils.LoadData("accuracy_monitor_setting", inference_name)
    _, monitor_setting = monitor_setting_data.load_data()
    return monitor_setting


def accuracyMonitor():
    while 1:
        query = {
            "match": {
                "status": "enable"
            }
        }
        result, items = utils.search_index("accuracy_monitor_setting", query)
        if result is False:
            logger.warning(items)
        else:
            producer = makeproducer()
            if producer is False:
                logger.warning('kafka.errors.NoBrokersAvailable: NoBrokersAvailable')
            else:
                for item in items:
                    inference_name = item['_id']
                    model_id = item['_source']['current_model']
                    drift_metric = item['_source']['drift_metric']
                    drift_measurement = item['_source']['drift_measurement']
                    atrisk_value = item['_source']['atrisk_value']
                    failing_value = item['_source']['failing_value']

                    try:
                        result, inference_info = utils.LoadData('accuracy_monitor_info',
                                                                f"{inference_name}_{model_id}").load_data()
                        model_type = inference_info['model_type']
                    except:
                        logger.warning(f'Name :{inference_name} does not have accuracy monitor info')
                        disable_monitor(inference_name)
                        continue

                    metric_value, count = get_accuracy_monitor(inference_name, model_id, drift_metric, model_type)
                    if count == 0 or count < 100:
                        accuracy_result = 'unknown'
                    elif metric_value == "N/A":
                        accuracy_result = "unknown"
                    else:
                        _, base_metric = utils.LoadData('accuracy_base_metric',
                                                        f"{inference_name}_{model_id}").load_data()
                        if drift_measurement == 'value':
                            value = utils.make_value(base_metric, drift_metric, metric_value)
                            if value == 'N/A':
                                return 'unknown'
                            if value > 0:
                                if value > failing_value:
                                    accuracy_result = 'failing'
                                elif value > atrisk_value:
                                    accuracy_result = 'atrisk'
                                else:
                                    accuracy_result = 'pass'
                            else:
                                accuracy_result = 'pass'
                        else:
                            percent = utils.make_percent(base_metric, drift_metric, metric_value)
                            if percent == 'N/A':
                                return 'unknown'
                            if percent < 0:
                                percent *= -1
                                if percent > failing_value:
                                    accuracy_result = 'failing'
                                elif percent > atrisk_value:
                                    accuracy_result = 'atrisk'
                                else:
                                    accuracy_result = 'pass'
                            else:
                                accuracy_result = 'pass'
                    try:
                        produceKafka(producer, {"inference_name": inference_name, "result": accuracy_result})
                    except:
                        logger.warning("producer is none")
                try:
                    producer.close()
                except:
                    logger.warning("producer error")
        time.sleep(30)


def disable_monitor(inference_name):
    result, message = utils.update_data('accuracy_monitor_setting', inference_name, {"status": "disable"})
    if result is False:
        return False
    return True


def enable_monitor(inference_name):
    result, message = utils.update_data('accuracy_monitor_setting', inference_name, {"status": "enable"})
    if result is False:
        return False
    return True


def update_association(inference_name, association_id):
    _, monitor_setting = utils.LoadData('accuracy_monitor_setting', inference_name).load_data()
    current_model = monitor_setting['current_model']
    _, inference_info = utils.LoadData('accuracy_monitor_info', f"{inference_name}_{current_model}").load_data()

    if association_id is None:
        result, message = utils.update_data('accuracy_monitor_info', f"{inference_name}_{current_model}",
                                            {"association_id": None, "association_index": None})
        if result is False:
            return False
    else:
        columns = inference_info["columns"]
        column_index = columns.index(association_id)
        result, message = utils.update_data('accuracy_monitor_info', f"{inference_name}_{current_model}",
                                            {"association_id": association_id, "association_index": column_index})
        if result is False:
            return False

    return True


############################################################
# Main
############################################################


if __name__ == '__main__':
    Process(target=accuracyMonitor).start()
    consumeKafka()

    # test
    # app.run(threaded=True, port=5001)
