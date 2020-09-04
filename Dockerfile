FROM ubuntu

RUN apt update 
RUN apt install -y python3-pip

COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
EXPOSE 5000

cmd ["app.py"]