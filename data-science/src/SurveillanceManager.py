from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Image,
)
import vertexai
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import re
from typing import List, Tuple, Dict
import numpy as np
import pickle as pkl
from ComputerVisionModel import ComputerVisionModel
from PromptModel import PromptModel
from AnalysisModel import AnalysisModel

class SurveillanceManager:
    def __init__(self, google_project: str, location: str, service_account_json_path: str):
        self.google_project = google_project
        self.google_location = location
        self.service_account_json_path = service_account_json_path

    @property
    def credentials(self) -> Credentials:
        # Search "How to Use Vertex AI API in Google Cloud" video in Youtube at timestamp 2:40 for tutorial how to download this json. provide the path of the downloaded json here.
        credentials = Credentials.from_service_account_file(
            self.service_account_json_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform'])  # Fixed url, dont change !
        if credentials.expired:
            credentials.refresh(Request())
        return credentials

    def init_vertex_ai(self):
        vertexai.init(project=self.google_project, location=self.google_location, credentials=self.credentials)

