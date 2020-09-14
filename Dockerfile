FROM python:3.7-slim

RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential libgl1-mesa-glx libgtk2.0-dev python3-distutils

# проверяем окружение python
RUN python3 --version
RUN pip3 --version

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN ls -la
#RUN mkdir app
COPY /frontend/* /app/
COPY /ml_server/* /app/

CMD cd /app && python app.py