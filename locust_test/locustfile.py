from locust import HttpUser, task, between

class DiabetesPredictorUser(HttpUser):
    wait_time = between(0.1, 0.5)
    host = "http://api.localhost:8081"

    @task
    def make_prediction(self):
        payload = {
          "race": "Caucasian", "gender": "Female", "age": "[70-80)",
          "time_in_hospital": 5, "num_lab_procedures": 40,
          "num_procedures": 1, "num_medications": 15,
          "diag_1": "250.83", "diag_2": "401", "diag_3": "250.01"
        }
        self.client.post("/predict", json=payload)
