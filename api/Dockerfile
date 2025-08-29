FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app


RUN pip install uv 

COPY ../pyproject.toml .

RUN uv venv
RUN uv sync 


COPY .. .

# Set PYTHONPATH so we can import from worker directory
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000","--reload"]