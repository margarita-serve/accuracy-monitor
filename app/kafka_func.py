import logging
import os
import time
from json import loads, dumps

import numpy as np
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer

from utils import LoadData, es_bulk

KSERVE_API_DEFAULT_KAFKA_ENDPOINT = os.environ.get('KSERVE_API_DEFAULT_KAFKA_ENDPOINT')

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger()


def makeproducer():
    try:
        producer = KafkaProducer(
            acks=1,
            compression_type='gzip',
            bootstrap_servers=[KSERVE_API_DEFAULT_KAFKA_ENDPOINT],
            value_serializer=lambda x: dumps(x).encode('utf-8')
        )
        return producer
    except Exception as err:
        logger.exception(err)
        time.sleep(60)
        makeproducer()


def produceKafka(producer, message):
    producer.send('accuracy-monitoring-data', value=message)
    producer.flush()


def consumeKafka():
    # topic, broker list
    try:
        consumer = KafkaConsumer(
            'logstash-inference-data',
            bootstrap_servers=[KSERVE_API_DEFAULT_KAFKA_ENDPOINT],
            group_id='accuracy-monitor',
            auto_offset_reset='latest',
            enable_auto_commit=True,
            # consumer_timeout_ms=1000,
            value_deserializer=lambda x: loads(x.decode('utf-8'))
        )
    except Exception as err:
        logger.exception(f"Kafka consumer Error: {err}")
        time.sleep(60)
        consumeKafka()

    # 접속확인
    if consumer.bootstrap_connected() is True:
        logger.info('Kafka consumer is running!')

    for message in consumer:
        try:
            if "tags" in message.value:
                logger.info(message.value['tags'])
            else:
                result, msg = trans_data(message.value)
                if result is False:
                    logger.warning(msg)
        except Exception as err:
            logger.exception(f'Kafka consumer error: {err}')
            continue


def trans_data(value):
    timestamp = value['timestamp']
    inference_id = value['inference_servicename']
    event_type = value['type']
    event_id = value['event_id']
    monitor_data = LoadData('accuracy_monitor_setting', inference_id)
    result, monitor = monitor_data.load_data()
    if result is False:
        return False, monitor
    if monitor['status'] == 'disable':
        return True, 'disable'

    current_model = monitor["current_model"]
    result, monitor_info = LoadData('accuracy_monitor_info', f"{inference_id}_{current_model}").load_data()
    if result is False:
        return False, monitor_info
    association_id = monitor_info["association_id"]

    count = 0
    bulk = []
    if event_type == 'request':
        instances = value['instances']
        df = pd.DataFrame(instances)
        trans_value = df.to_dict('records')

        for i in range(len(instances)):
            if value.get(association_id):
                association_value = value[association_id]
                # trans_value = {v: k for v, k in enumerate(instances[i])}
                doc = {
                    "event_id": event_id,
                    "timestamp": timestamp,
                    "inference_id": inference_id,
                    "instance": trans_value[i],
                    association_id: association_value[i],
                    "model_history_id": current_model
                }
            else:
                # trans_value = {v: k for v, k in enumerate(instances[i])}
                doc = {
                    "event_id": event_id,
                    "timestamp": timestamp,
                    "inference_id": inference_id,
                    "instance": trans_value[i],
                    "model_history_id": current_model
                }

            bulk.append({
                "_op_type": "update",
                "_index": f"accuracy_inference_data_{inference_id}",
                "_type": "_doc",
                "_id": f"{event_id}-{count}",
                "doc": doc,
                "doc_as_upsert": True
            })
            # result, message = upsert_doc(f"accuracy_inference_data_{inference_id}", f"{event_id}-{count}", doc)
            count += 1
        result, message, _ = es_bulk(bulk)
        return result, message
    elif event_type == 'response':
        predictions = value['predictions']
        for prediction in predictions:
            if len(np.shape(predictions)) == 1:
                trans_value = {"0": prediction}
            else:
                trans_value = {v: k for v, k in enumerate(prediction)}
            doc = {
                "event_id": event_id,
                "timestamp": timestamp,
                "inference_id": inference_id,
                "prediction": trans_value,
                "model_history_id": monitor['current_model']
            }
            bulk.append({
                "_op_type": "update",
                "_index": f"accuracy_inference_data_{inference_id}",
                "_type": "_doc",
                "_id": f"{event_id}-{count}",
                "doc": doc,
                "doc_as_upsert": True
            })
            # result, message = upsert_doc(f"accuracy_inference_data_{inference_id}", f"{event_id}-{count}", doc)
            count += 1
        result, message, _ = es_bulk(bulk)
        return result, message
    else:
        return False, f'type {event_type} is wrong'
