FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app


RUN pip install uv 

COPY ../pyproject.toml .

RUN uv venv
RUN uv sync 

 
# Copy the entire project
COPY .. .


ENV PYTHONPATH=/app


CMD ["uv","run","worker/run_worker.py"]