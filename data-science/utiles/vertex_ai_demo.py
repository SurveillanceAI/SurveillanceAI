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

# Search "How to Use Vertex AI API in Google Cloud" video in Youtube at timestamp 2:40 for tutorial how to download this json. provide the path of the downloaded json here.
key_path = "/home/yonatan.r/Downloads/astral-sunbeam-443219-p5-016df1c01623.json" 
credentials = Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform']) # Fixed url, dont change !
if credentials.expired:
    credentials.refresh(Request())

# project = Project ID of your GCP Project
# location = location of your project, search for the Vertex AI tab in your GCP to get the locatino
vertexai.init(project="astral-sunbeam-443219-p5", location="us-central1", credentials=credentials)
MODEL_ID = "gemini-1.5-flash-002" # any google method, search here for more models https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-pro-vision?inv=1&invt=AblIyQ&project=datamind-production

# system_instruction = Explation to the model to exaplin who he is, before it is given a specific task.
model = GenerativeModel(
    MODEL_ID,
    system_instruction=[
        "You are a highly skilled security guard working at a retail store.",
        "Your main responsibility is to monitor customer behavior and identify potential shoplifting activities.",
    ],
)

# Set model parameters
# need to investigate more about what each parameter does and its effect 
generation_config = GenerationConfig(
    temperature=0.9,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

# Set safety settings.
# Dont know what does it mean, need to investigate
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

prompt = """
You are a highly skilled security guard working at a retail store. Your main responsibility is to monitor customer behavior and identify potential shoplifting activities. Pay close attention to behaviors such as: 

1. Unusual or suspicious movements (e.g., concealing items in clothing or bags).
2. Avoiding staff or surveillance cameras.
3. Repeatedly entering and exiting the store without making purchases.
4. Carrying empty bags into the store and leaving with them full.
5. Lingering in certain areas of the store without clear purpose.

Whenever you detect a suspicious activity, describe the specific behavior, the location in the store, and why you believe it could indicate shoplifting. Be polite and professional in your observations, focusing solely on identifying and reporting potential incidents. Remember to prioritize accuracy and fairness to ensure no false accusations.
"""

#uri is the place of the video in your bucket
video_file = Part.from_uri(uri="gs://example_bucket_final_project/43dd8387-28ad-4a64-bda1-9c566c526b82.MP4", mime_type="video/mp4")

# Set contents to send to the model
contents = [video_file, prompt]

# Prompt the model to generate content
response = model.generate_content(
    contents,
    generation_config=generation_config,
    safety_settings=safety_settings,
)
print(response.text)
