FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app


RUN pip install uv 

COPY ../pyproject.toml .

RUN uv venv
RUN uv sync 

 
# Copy the entire project
COPY ../api .
COPY ../tests .
COPY ../worker .
COPY ../.env .
COPY ../README.md .
COPY ../run_local.sh . 
COPY ../run_tests.sh . 
COPY ../__init__.py . 


ENV PYTHONPATH=/app


CMD ["uv","run","worker/run_worker.py"]