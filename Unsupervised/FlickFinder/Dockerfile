FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
EXPOSE 5001
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5001", "--server.address=0.0.0.0"]

