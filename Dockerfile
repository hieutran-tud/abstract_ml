FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY notebooks ./notebooks

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[notebooks]"

EXPOSE 8888

CMD ["jupyter", "lab", "notebooks", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--ServerApp.token="]
