FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app


RUN pip install uv 

COPY ../pyproject.toml .

RUN uv venv
RUN uv sync 


COPY ../api .
COPY ../tests .
COPY ../worker .
COPY ../.env .
COPY ../README.md .
COPY ../run_local.sh . 
COPY ../run_tests.sh . 
COPY ../__init__.py . 




# Set PYTHONPATH so we can import from worker directory
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000","--reload"]