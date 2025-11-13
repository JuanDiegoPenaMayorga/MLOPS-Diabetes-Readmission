# End-to-End MLOps: Diabetes Patient Readmission Prediction

This repository contains a full-stack, end-to-end MLOps system designed to train, serve, and monitor a machine learning model that predicts hospital readmission for diabetes patients. It showcases a production-style architecture separating the MLOps infrastructure (training, orchestration, governance) from the application serving layer.

---

## Architecture Overview

The system employs a hybrid architecture to decouple the MLOps backend from the user-facing services:

-   **MLOps Infrastructure (Docker Compose):**
    -   **Orchestration:** `Apache Airflow` manages all data and training pipelines.
    -   **Experiment Tracking:** `MLflow Tracking` records parameters, metrics, and models for every training run.
    -   **Model Governance:** `MLflow Model Registry` versions the models and manages their stages (Staging, Production).
    -   **Artifact Storage:** `MinIO` provides an S3-compatible object store for model artifacts.
    -   **Data Storage:** `PostgreSQL` instances serve as backends for Airflow, MLflow, and for storing raw and cleaned data.

-   **Serving Infrastructure (Kubernetes via k3d):**
    -   **Inference API:** A `FastAPI` microservice loads the current production model from MLflow and serves predictions via a REST API.
    -   **Web UI:** A `Streamlit` application provides a user-friendly interface to interact with the prediction API.
    -   **Observability Stack:** `Prometheus` scrapes metrics from the API, and `Grafana` provides real-time monitoring dashboards.
    -   **Load Testing:** `Locust` is used to simulate user traffic and stress-test the API, validating its performance under load.

![Project Architecture](https://i.imgur.com/e4sBgtZ.png)  

---

## Features

-   **Automated Pipelines:** Airflow DAGs automate data ingestion and incremental model training.
-   **True Incremental Learning:** The main training pipeline (`03_warm_start_learning`) updates the model with new data batches without retraining from scratch, demonstrating efficient continuous learning.
-   **Decoupled Model Serving:** The API dynamically loads whichever model version is flagged as "Production" in MLflow, allowing for model updates without any code changes or downtime.
-   **Containerized & Scalable:** The entire system runs in containers. The serving components are deployed on Kubernetes, ready to be scaled horizontally.
-   **Real-time Observability:** Live monitoring of API throughput, latency, and error rates using Prometheus and Grafana.

---

## Getting Started

This guide assumes a **Debian-based environment (like Ubuntu 24.04)** with `sudo` access and at least 12 GB RAM.

### 1. Prerequisites

First, install the necessary command-line tools.

```bash
# Install system dependencies
sudo apt update && sudo apt install -y curl docker-compose-plugin pipx

# Setup pipx (for safe CLI tool installation)
pipx ensurepath
# IMPORTANT: Close and reopen your terminal after this step.

# Install Kubernetes tools
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
sudo snap install kubectl --classic
```

### 2. Initial Local Setup

If you have cloned this repository, navigate into its directory. The following steps prepare the local environment.

```bash
# Add local domains for easier access. This is a one-time setup.
echo "127.0.0.1   api.localhost ui.localhost" | sudo tee -a /etc/hosts

# Ensure the Docker daemon is running
sudo systemctl start docker
```

### 3. Launching the System (Quick Start)

This process will launch both the Docker Compose infrastructure and the Kubernetes applications from a clean state. Run these commands from the project root.

```bash
# 1. Start the MLOps Backend Services (Airflow, MLflow, etc.)
docker-compose up -d

# 2. Initialize the Airflow Database
# Wait a minute for the postgres container to be healthy, then run:
docker-compose run --rm airflow-webserver db init

# 3. Create the Kubernetes Cluster and Connect It to the Docker Network
# This command ensures K8s pods can communicate with Docker Compose services.
NETWORK_NAME=$(docker network ls | grep proyecto3_default | awk '{print $2}')
k3d cluster create mlops --api-port 6443 -p "8081:80@loadbalancer" --network \$NETWORK_NAME
k3d kubeconfig merge mlops --kubeconfig-merge-default

# 4. Import Application Images into the Cluster
(cd api && docker build -t diabetes-api:latest . && k3d image import diabetes-api:latest -c mlops)
(cd streamlit_ui && docker build -t diabetes-ui:latest . && k3d image import diabetes-ui:latest -c mlops)

# 5. Deploy Applications and Monitoring to Kubernetes
kubectl apply -f ./api/k8s/
kubectl apply -f ./streamlit_ui/k8s/
kubectl apply -f ./monitoring/prometheus.yaml
kubectl apply -f ./monitoring/grafana.yaml
```

Wait for all pods to be in the `Running` state (check with `kubectl get pods -w` and `kubectl get pods -n monitoring -w`). Once ready, you can access the system's UIs.

---

## Access Points & UIs

-   **Airflow UI:** `http://localhost:8080` (User: `Admin`, Pass: `SuperSecret`)
-   **MLflow UI:** `http://localhost:5000`
-   **MinIO Console:** `http://localhost:9001` (User: `Admin`, Pass: `SuperSecret`)
-   **Project Web UI:** `http://ui.localhost:8081`
-   **Inference API:** `http://api.localhost:8081`
-   **Grafana UI:** `http://localhost:3000` (Requires port-forward. User: `admin`, Pass: `admin`)

---

## MLOps Workflow Demonstration

Follow these steps to run a full training and deployment cycle.

### Step 1: Ingest Data
1.  Navigate to the **Airflow UI** (`localhost:8080`).
2.  Enable and trigger the `01_ingest_data_dag` DAG.
3.  This pipeline seeds the `RAW_DATA` PostgreSQL database, making data available for training.

### Step 2: Train & Register the Model
1.  In Airflow, enable and trigger the `03_warm_start_training` DAG.
2.  Run it multiple times. Each run loads the previously trained model, updates it with a new batch of 15,000 records, and logs the new version to MLflow.
3.  Navigate to the **MLflow UI** (`localhost:5000`).
4.  Open the `diabetes_warm_start_training` experiment, find the run with the best `batch_accuracy`, and register its model. Name it **`diabetes-readmission-predictor`**.
5.  In the "Models" tab, find this model and transition its latest version to the **`Production`** stage.

### Step 3: Test the Production Model
1.  Navigate to the **Project Web UI** (`ui.localhost:8081`).
2.  Fill in the form with patient data and get a prediction. The UI is calling the API that is now serving your newly promoted production model.
3.  Alternatively, test the API directly:
    ```bash
    curl -X POST "http://api.localhost:8081/predict" \
    -H "Content-Type: application/json" \
    -d '{
      "race": "Caucasian", "gender": "Female", "age": "[70-80)",
      "time_in_hospital": 5, "num_lab_procedures": 40,
      "num_procedures": 1, "num_medications": 15,
      "diag_1": "250.83", "diag_2": "401", "diag_3": "250.01"
    }'
    ```

---

## Observability and Load Testing

### 1. Monitor API Performance with Grafana
1.  Open a dedicated terminal to forward the Grafana port:
    ```bash
    kubectl port-forward --address 0.0.0.0 svc/grafana-service 3000:3000 -n monitoring
    ```
2.  Access Grafana at `http://localhost:3000` (or `http://YOUR_PC_IP:3000` from an external device).
3.  Add the Prometheus data source with the URL: `http://prometheus-simple-service.monitoring:9090`.
4.  Create a dashboard with queries like `sum(rate(http_requests_total{handler="/predict"}[1m]))` to monitor throughput.

### 2. Simulate User Traffic with Locust
1.  Install Locust (one-time setup):
    ```bash
    pipx install locust
    ```
2.  Run the load test:
    ```bash
    cd locust_test/
    locust -f locustfile.py
    ```
3.  Open the Locust UI at `http://localhost:8089`, start a swarm against the `http://api.localhost:8081` host, and observe the metrics on your Grafana dashboard in real-time.

---

## Shutdown & Cleanup

When you're finished, shut down all services to free up system resources.

```bash
# 1. Delete the Kubernetes cluster and all its resources
k3d cluster delete mlops

# 2. Stop and remove all Docker Compose services and networks
docker-compose down

# (Optional) For a complete cleanup that deletes all data volumes:
# docker-compose down -v
```
---

## License

This project is distributed under the **MIT License**.  
You are free to use, modify, and distribute it with proper attribution.

---

**Developed by [Juan Diego Peña Mayorga](https://www.linkedin.com/in/jdpm97/)**  
_Bogotá, Colombia_  
_Focused on automation, AI and system integration._

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Juan%20Diego%20Peña-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/jdpm97/)
[![GitHub](https://img.shields.io/badge/GitHub-JuanDiegoPenaMayorga-181717?style=for-the-badge&logo=github)](https://github.com/JuanDiegoPenaMayorga)
[![Email](https://img.shields.io/badge/Email-JuanDiegoPena%40hotmail.com-red?style=for-the-badge&logo=gmail)](mailto:JuanDiegoPena@hotmail.com)
