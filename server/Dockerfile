# Image for the yearly pipeline (Fargate task). The API does not use this -
# it runs on Lambda from a zip (deploy/deploy.sh).
FROM python:3.12-slim

WORKDIR /app/server

COPY requirements.txt requirements-pipeline.txt ./
RUN pip install --no-cache-dir -r requirements-pipeline.txt boto3==1.43.111

COPY app ./app
COPY scripts ./scripts
# Static inputs that never change between years. Results and the model are
# pulled from S3 at run time so the image does not go stale.
COPY data/himag_all.json data/window_sizes.json ./data/

CMD ["python", "-m", "scripts.cloud_pipeline"]
