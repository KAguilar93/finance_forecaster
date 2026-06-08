FROM python:3.11-slim-bookworm

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

EXPOSE 8080

ENTRYPOINT ["python", "-u", "-m", "finance_forecaster.predict_model"]
