# FROM python:3.10-slim
FROM python:3.9

WORKDIR /app

RUN pip install streamlit requests

COPY . /app

EXPOSE 8501

# ENTRYPOINT ["streamlit", "run", "app.py"]

# Command to run the Streamlit app
CMD ["streamlit", "run", "code/deployment/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]