FROM tensorflow/tensorflow
RUN pip install --upgrade pip
RUN pip install matplotlib
RUN apt-get -y update
RUN apt-get -y install python-tk
