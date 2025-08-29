FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app


RUN pip install uv 

COPY ../pyproject.toml .
# COPY ../uv.lock .

RUN uv sync 
#--locked

# RUN uv venv /venv && uv sync --locked
 
# Copy the entire project
COPY .. .


ENV PYTHONPATH=/app


CMD ["uv","run","worker/run_worker.py"]