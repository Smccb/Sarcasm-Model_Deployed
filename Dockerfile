
FROM python:3.10
#Labels as key value pair
LABEL Maintainer="me"
# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /
#to COPY the remote file at working directory in container

COPY requirements.txt ./
COPY Flask.py ./
#COPY Tokenizer2.pickle ./
COPY templates/ ./templates
# COPY model_cudnn_lstm_weights2.h5 ./
# COPY model_cudnn_lstm_architecture2.joblib ./

COPY non_GPU_model_Tokenizer.pickle ./
COPY non_GPU_model_trained.h5 ./

# Now the structure looks like this '/usr/app/src/test.py'
#CMD instruction should be used to run the software
#contained by your image, along with any arguments.
RUN pip install flask
RUN pip install flask-restful
RUN pip install -r requirements.txt
#Run pip install tensorflow 2.10
Run pip install keras

EXPOSE 5000
CMD [ "python", "./Flask.py"]
