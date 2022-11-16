# Accuracy-Monitor-API Server
모델의 Accuracy를 Monitoring 하는 API 서버.

## 실행 환경설정
```shell
# 모델과 데이터셋이 저장된 storage 정보 ( default = minio )
export KSERVE_API_DEFAULT_STORAGE_ENDPOINT="192.168.88.154:9000"
export KSERVE_API_DEFAULT_AWS_ACCESS_KEY_ID=bWluaW9hZG1pbg==
export KSERVE_API_DEFAULT_AWS_SECRET_ACCESS_KEY=bWluaW9hZG1pbg==

# Prediction 정보 를 받고 Drift 모니터링 결과를 전달할 kafka 위치
export KSERVE_API_DEFAULT_KAFKA_ENDPOINT="192.168.88.151:32000"

# inference의 모델과 데이터세트 정보 및 예측정보를 저장 위치 ( default = elasticsearch )
export KSERVE_API_DEFAULT_DATABASE_ENDPOINT="http://192.168.88.151:30092"
```


## 로컬환경 실행
```shell
RESTAPI server
$ ./gunicorn.sh

monitoring server
$ python main.py
```

## Docker실행
```shell
docker run --env-file ./.env -p 8003:8003 192.168.88.155/koreserve/accuracy-monitor:{tag}
```

## REST-API
- For swagger document you have to request root directory(/)
```shell
http://yourhostname:8003/
```