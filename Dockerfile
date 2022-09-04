FROM python:3.9

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r  requirements.txt 

ENV PORT=8051

COPY . .

CMD streamlit run main.py --server.port=${PORT}  --browser.serverAddress="0.0.0.0"
