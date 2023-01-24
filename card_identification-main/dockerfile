FROM python:3.8.5

RUN pip install virtualenv
WORKDIR /app                                                                

COPY ["requirements.txt", "./"]

RUN pip install -r /app/requirements.txt

COPY ["*.py", "*.h5", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]