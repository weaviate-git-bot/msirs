FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get -y install curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="$PATH:/root/.cargo/bin"

COPY senet-docker/requirements.txt .
COPY senet-docker/fullAdaptedSENetNetmodel.keras .
COPY senet_model.py .
RUN pip3 install -r requirements.txt

COPY senet-docker/ .

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8081"]
