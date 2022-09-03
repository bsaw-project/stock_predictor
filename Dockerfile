FROM python:3.9

EXPOSE 8051

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]