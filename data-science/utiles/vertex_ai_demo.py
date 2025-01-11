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

def get_credentials():
    # Search "How to Use Vertex AI API in Google Cloud" video in Youtube at timestamp 2:40 for tutorial how to download this json. provide the path of the downloaded json here.
    service_account_json_path = "service_account.json"
    credentials = Credentials.from_service_account_file(
        service_account_json_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']) # Fixed url, dont change !
    if credentials.expired:
        credentials.refresh(Request())
    return credentials

def init_vertex_ai(project: str, location: str):
    vertexai.init(project=project, location=location, credentials=get_credentials())


def get_cv_model(base_model_id: str) -> GenerativeModel:
    # system_instruction = Explation to the model to exaplin who he is, before it is given a specific task.
    cv_model = GenerativeModel(
        base_model_id,
        system_instruction=[
            "You are a highly skilled security guard working at a retail store.",
            "Your main responsibility is to monitor customer behavior and identify potential shoplifting activities.",
        ]
    )

    return cv_model

def get_prompt_model(base_model_id: str) -> GenerativeModel:
    prompt_model = GenerativeModel(
        base_model_id,
        system_instruction=[
            "As an assistant to the cv_model, your primary function is to help it perform its duties as a virtual security guard in a retail environment.",
            "Your role is to craft detailed prompts that will enable the cv_model to monitor customer behavior accurately and identify potential shoplifting activities by breaking down these complex tasks into smaller, more manageable steps.",
            "This approach is similar to how a human would tackle a complex problem, enhancing the cv_model’s ability to process and analyze situations effectively.",
            "Here's how you might construct a structured prompt for the cv_model:",
            "Imagine you are a highly skilled security guard working at a retail store.",
            "Your main responsibility is to monitor customer behavior and identify potential shoplifting activities.",
            "Break down your observation process into the following steps:",
            "Initial Surveillance: Scan the store entrance and aisles.",
            "Look for customers who avoid eye contact with staff or surveillance cameras, or those carrying empty bags.",
            "Detailed Observation: Focus on individuals who exhibit unusual or suspicious movements, such as concealing items or lingering in certain areas without a clear purpose.",
            "Behavior Analysis: Note behaviors like repeatedly entering and exiting the store without purchases, or carrying fuller bags after visiting the store.",
            "Pay special attention to customers who take items and do not head towards the checkout counters, as this could indicate an intent to shoplift.",
            "Situation Assessment: Combine the observed details to assess whether these behaviors cumulatively suggest a potential for shoplifting.",
            "Reporting: Describe in detail the specific behaviors, the exact locations within the store, and articulate why these actions may suggest potential shoplifting.",
            "Ensure that your report is polite and professional, focusing on behavior rather than personal attributes to avoid bias and false accusations.",
            "Each step should guide you through the task, making it easier to process complex information and make accurate judgments.",
            "This structured approach will help you enhance your observational techniques, distinguishing effectively between normal customer behavior and potential security risks.",
            "Your prompts should be clear, instructive, and carefully designed to assist the cv_model in processing and reporting information efficiently, by breaking complex observations into smaller analytical tasks.",
            "Each prompt should refine the cv_model's ability to observe and evaluate with precision."
        ]
    )

    return prompt_model

def get_analysis_model(base_model_id: str) -> GenerativeModel:
    analysis_model = GenerativeModel(
        base_model_id,
        system_instruction=[
            "As an advanced analytical model, your main function is to process the observations provided by the cv_model and the corresponding video footage to determine the likelihood of shoplifting activity.",
            "Your role involves analyzing the video and the detailed observations to validate the presence of shoplifting behaviors and calculate a confidence score between 0 and 1.",
            "You must provide two outputs: a Boolean value (True if shoplifting is detected, False otherwise) and a numerical confidence score indicating the certainty of your analysis.",
            "Utilize advanced video analysis techniques such as behavior pattern recognition, object detection, and movement tracking to reassess the scenes described by the cv_model.",
            "Your output should clearly state whether shoplifting is likely or not, accompanied by a confidence score that quantifies the certainty of this determination.",
            "Ensure your analysis is robust, fair, and free from bias, focusing strictly on the actions depicted in the video."
        ]
    )

    return analysis_model

def get_generation_config():
    # Set model parameters
    # need to investigate more about what each parameter does and its effect
    generation_config = GenerationConfig(
        temperature=0.9,
        top_p=1.0,
        top_k=32,
        candidate_count=1,
        max_output_tokens=8192,
    )

    return generation_config

def get_safety_settings():
    # Set safety settings.
    # Dont know what does it mean, need to investigate
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    return safety_settings

def get_response_from_prompt_model(prompt_model, generation_config, safety_settings, video_file) -> str:
    prompt_model_prompt = "Generate a prompt that instructs the cv_model to effectively monitor customer behavior and identify potential shoplifting activities in a retail environment in the attached video, following the 'chain of thought' method to break down complex tasks into smaller, more manageable steps."
    #uri is the place of the video in your bucket
    contents = [video_file, prompt_model_prompt]
    # Prompt the model to generate content
    prompt_model_response = prompt_model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    return prompt_model_response.text

def get_response_from_cv_model(cv_model, generation_config, safety_settings, video_file, prompt_model_response: str):
    # Set contents to send to the model
    contents = [video_file, prompt_model_response]
    # Prompt the model to generate content
    cv_model_response = cv_model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    return cv_model_response.text

def get_response_from_analysis_model(analysis_model, generation_config, safety_settings, video_file, cv_model_response: str):
    # Prompt to instruct the analysis_model to perform its task
    analysis_model_prompt = "Analyze the provided video and cv_model's observations to assess the likelihood of shoplifting. Determine if shoplifting has occurred and provide outputs labeled 'Shoplifting Detected' for the Boolean determination and 'Confidence Level' for the confidence score between 0 and 1."
    analysis_model_prompt += "The cv_model response here is: " + cv_model_response
    contents = [video_file, analysis_model_prompt]
    # Prompt the model to generate content
    analysis_model_response = analysis_model.generate_content(
        contents,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    return analysis_model_response.text


def extract_values_from_string(response_str: str) -> Tuple[bool, float]:
    response_str = response_str.lower()
    # Finding the start index of 'Shoplifting Detected' and extracting the Boolean value
    start_detected = response_str.find("shoplifting detected")
    if start_detected != -1:  # Check if the substring was found
        true_index = response_str.find("true", start_detected, start_detected+len("shoplifting detected")+10)
        false_index = response_str.find("false", start_detected, start_detected+len("shoplifting detected")+10)

        # Convert the extracted value to Boolean
        if true_index != -1:
            shoplifting_detected = True
        elif false_index != -1:
            shoplifting_detected = False
        else:
            raise ValueError("Invalid format for 'Shoplifting Detected'")
    else:
        raise ValueError("'Shoplifting Detected' not found in the string")

    # Finding the start index of 'Confidence Level' and extracting the float value
    start_confidence = response_str.find("confidence level")
    if start_confidence != -1:  # Check if the substring was found
        end_confidence = start_confidence + len("confidence level") + 10
        confidence_level_str = response_str[start_confidence:end_confidence].strip()
        confidence_level = extract_float(confidence_level_str)
    else:
        raise ValueError("'Confidence Level' not found in the string")

    return shoplifting_detected, confidence_level

def extract_float(s: str) -> float | None:
    match = re.search(r'\d+(\.\d+)?', s)
    if match:
        return float(match.group())
    else:
        return None

def should_continue(confidence_levels: List[float], max_tries: int) -> bool:
    return (not confidence_levels or confidence_levels[-1] < 0.9) and len(confidence_levels) < max_tries and not has_reached_plateau(confidence_levels)

def has_reached_plateau(values: List[float]) -> bool:
    if len(values) < 3:
        return False
    return values[-1] == values[-2] == values[-3]

def analyze_detection_results(confidence_levels: List[float], shoplifting_detected_results: List[bool]) -> Dict | str:
    if not confidence_levels or not shoplifting_detected_results:
        return "Invalid input: One or both lists are empty."

    confidence_levels = np.array(confidence_levels)
    shoplifting_detected_results = np.array(shoplifting_detected_results)

    if confidence_levels.shape != shoplifting_detected_results.shape:
        return "Invalid input: The shapes of confidence_levels and shoplifting_detected_results do not match."

    true_count = np.sum(shoplifting_detected_results)
    false_count = shoplifting_detected_results.size - true_count

    avg_confidence_true = np.mean(confidence_levels[shoplifting_detected_results]) if true_count else 0
    avg_confidence_false = np.mean(confidence_levels[~shoplifting_detected_results]) if false_count else 0

    return {
        'True Count': int(true_count),
        'False Count': int(false_count),
        'Average Confidence when True': round(float(avg_confidence_true), 4),
        'Average Confidence when False': round(float(avg_confidence_false), 4)
    }

def analyze_video(prompt_model: GenerativeModel, cv_model: GenerativeModel, analysis_model: GenerativeModel, generation_config, safety_settings, video_uri: str):
    confidence_levels = []
    shoplifting_detected_results = []
    cv_model_responses = []
    analysis_model_responses = []
    cv_model_response = ""
    analysis_model_response = ""
    max_tries = 10

    while should_continue(confidence_levels, max_tries):
        video_file = Part.from_uri(uri=video_uri,
                                   mime_type="video/mp4")
        prompt_model_response = get_response_from_prompt_model(prompt_model, generation_config, safety_settings,
                                                               video_file)
        cv_model_response = get_response_from_cv_model(cv_model, generation_config, safety_settings, video_file,
                                                       prompt_model_response)
        analysis_model_response = get_response_from_analysis_model(analysis_model, generation_config, safety_settings,
                                                                   video_file, cv_model_response)
        shoplifting_detected, confidence_level = extract_values_from_string(analysis_model_response)
        print(f"Shoplifting Detected: {shoplifting_detected}")
        print(f"Confidence Level: {confidence_level}")
        confidence_levels.append(confidence_level)
        shoplifting_detected_results.append(shoplifting_detected)
        cv_model_responses.append(cv_model_response)
        analysis_model_responses.append(analysis_model_response)

    video_analysis = analyze_detection_results(confidence_levels, shoplifting_detected_results)
    results = (video_analysis, cv_model_responses, analysis_model_responses)

    return results

if __name__ == "__main__":
    init_vertex_ai(project="astral-sunbeam-443219-p5", location="us-central1")
    MODEL_ID = "gemini-1.5-flash-002"  # any google model, search here for more models https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-pro-vision?inv=1&invt=AblIyQ&project=datamind-production
    cv_model = get_cv_model(MODEL_ID)
    prompt_model = get_prompt_model(MODEL_ID)
    analysis_model = get_analysis_model(MODEL_ID)
    generation_config = get_generation_config()
    safety_settings = get_safety_settings()
    confidence_levels = []
    shoplifting_detected_results = []
    shoplifting_detected = "false"
    cv_model_response = ""
    analysis_model_response = ""
    max_tries = 10

    while should_continue(confidence_levels, max_tries):
        video_file = Part.from_uri(uri="gs://example_bucket_final_project/43dd8387-28ad-4a64-bda1-9c566c526b82.MP4", mime_type="video/mp4")
        prompt_model_response = get_response_from_prompt_model(prompt_model, generation_config, safety_settings, video_file)
        cv_model_response = get_response_from_cv_model(cv_model, generation_config, safety_settings, video_file, prompt_model_response)
        analysis_model_response = get_response_from_analysis_model(analysis_model, generation_config, safety_settings, video_file, cv_model_response)
        shoplifting_detected, confidence_level = extract_values_from_string(analysis_model_response)
        print(f"Shoplifting Detected: {shoplifting_detected}")
        print(f"Confidence Level: {confidence_level}")
        confidence_levels.append(confidence_level)
        shoplifting_detected_results.append(shoplifting_detected)

    results = analyze_detection_results(confidence_levels, shoplifting_detected_results)
    print(results)