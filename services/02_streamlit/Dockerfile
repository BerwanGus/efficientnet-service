FROM python:3.9-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./app /app/

ENTRYPOINT [ "streamlit", "run", "Dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8001" ]
