# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend/ ./

RUN npm run build

# Stage 2: Setup Python backend
FROM python:3.10-slim

RUN apt-get update && apt-get install -y git

ENV CONTAINER_HOME=/var/www
ARG SPARK_CLIENT_PIP_SPEC=infosci-spark-client
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG INSTALL_SPARK_CLIENT=false

WORKDIR $CONTAINER_HOME

COPY requirements.txt $CONTAINER_HOME/requirements.txt
RUN pip install --no-cache-dir -r $CONTAINER_HOME/requirements.txt
RUN if [ "$INSTALL_SPARK_CLIENT" = "true" ]; then \
      if [ -n "$PIP_INDEX_URL" ]; then \
        PIP_INDEX_URL="$PIP_INDEX_URL" PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL" \
        pip install --no-cache-dir "$SPARK_CLIENT_PIP_SPEC"; \
      elif [ -n "$PIP_EXTRA_INDEX_URL" ]; then \
        PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL" \
        pip install --no-cache-dir "$SPARK_CLIENT_PIP_SPEC"; \
      else \
        pip install --no-cache-dir "$SPARK_CLIENT_PIP_SPEC"; \
      fi; \
    else \
      echo "Skipping Spark client install; set INSTALL_SPARK_CLIENT=true and provide a package source to enable it."; \
    fi

COPY src/ $CONTAINER_HOME/src/
COPY data/ $CONTAINER_HOME/data/

COPY --from=frontend-build /app/frontend/dist $CONTAINER_HOME/frontend/dist

CMD ["gunicorn", "--chdir", "src", "app:app", "--bind", "0.0.0.0:5000"]
