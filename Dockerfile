FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src ./src
COPY examples ./examples
COPY eval ./eval

RUN pip install --no-cache-dir -e .[api]

ENV IDP_STORAGE_DIR=/data/idp
VOLUME ["/data/idp"]

EXPOSE 8000
CMD ["uvicorn", "examples.api:app", "--host", "0.0.0.0", "--port", "8000"]
