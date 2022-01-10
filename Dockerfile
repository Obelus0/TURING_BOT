FROM python:3.7
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY requirements2.txt ./requirements2.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip3 install -r requirements.txt --timeout 5000
RUN pip3 install -r requirements2.txt --timeout 5000
COPY . .
CMD streamlit run app.py
