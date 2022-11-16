FROM python:3.9-slim-buster

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY . /app
RUN chmod +x /app/gunicorn.sh
EXPOSE 8003

ENV PYTHONUNBUFFERED=0
#ENTRYPOINT [ "sh","/app/gunicorn.sh"]