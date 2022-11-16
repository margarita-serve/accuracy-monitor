from utils import msearch_query, es_bulk
from dateutil import parser


def upload_data(match_df, inference_name, association_id, association_column, association_index, actual_response,
                positive_class=False, negative_class=False, binary_threshold=None):
    df = match_df.drop_duplicates([association_column], keep='last')
    query = []
    update_date = set()
    if association_index is not None:
        for _, row in df.iterrows():
            query.append({'index': f'accuracy_inference_data_{inference_name}'})
            # query.append({'index': f'inference_org_{inference_name}'})
            query.append(
                {"query": {"match": {f"instance.{association_index}": f"{row[association_id]}"}},
                 "sort": {"timestamp": {"order": "desc"}}})

    else:
        for _, row in df.iterrows():
            query.append({'index': f'accuracy_inference_data_{inference_name}'})
            # query.append({'index': f'inference_org_{inference_name}'})
            query.append({"query": {"match": {f"{association_id}": f"{row[association_column]}"}},
                          "sort": {"timestamp": {"order": "desc"}}})

    result, value = msearch_query(query)
    if result is False:
        return False, value, ''
    count = 0
    bulk = []
    for v in value['responses']:
        if "error" in v:
            return False, v["status"], ''
        if len(v['hits']['hits']) == 0:
            count += 1
        elif 'prediction' in v['hits']['hits'][0]['_source']:
            predict_value = v['hits']['hits'][0]['_source']['prediction']['0']
            model_id = v['hits']['hits'][0]['_source']['model_history_id']
            if binary_threshold:
                # if type(df.iloc[count][actual_response].tolist()) is int:
                #     positive_class = int(1)
                #     negative_class = int(0)
                # elif type(df.iloc[count][actual_response].tolist()) is float:
                #     positive_class = float(1)
                #     negative_class = float(0)

                if predict_value >= binary_threshold:
                    predict_value = 1
                else:
                    predict_value = 0
            doc = {
                "actual_value": df.iloc[count][actual_response],
                "predict_value": predict_value,
                "timestamp": v['hits']['hits'][0]['_source']['timestamp'],
                "inference_id": v['hits']['hits'][0]['_source']['inference_id'],
                "association_id": df.iloc[count][association_column]
            }
            bulk.append({
                "_id": df.iloc[count][association_column],
                "_index": f"accuracy_match_data_{inference_name}_{model_id}",
                "_source": doc
            })
            parser_date = parser.parse(v['hits']['hits'][0]['_source']['timestamp'])
            hour_date = parser_date.strftime('%Y-%m-%dT%H:00:00')
            m_data = model_id + "," + hour_date
            update_date.add(m_data)
            count += 1
        else:
            count += 1

    result, message, match_count = es_bulk(bulk)
    if result is False:
        return result, message, ''
    date_list = list(update_date)

    return result, message, date_list, match_count
