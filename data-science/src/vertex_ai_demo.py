from vertexai.generative_models import Part
import vertexai
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from typing import List, Tuple, Dict
import numpy as np
import pickle
from ComputerVisionModel import ComputerVisionModel
from PromptModel import PromptModel
from AnalysisModel import AnalysisModel
import datetime
from google.cloud import storage
import os
import pandas as pd

def get_credentials() -> Credentials:
    # Search "How to Use Vertex AI API in Google Cloud" video in Youtube at timestamp 2:40 for tutorial how to download this json. provide the path of the downloaded json here.
    service_account_json_path = ""
    credentials = Credentials.from_service_account_file(
        service_account_json_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']) # Fixed url, dont change !
    if credentials.expired:
        credentials.refresh(Request())
    return credentials

def init_vertex_ai(project: str, location: str):
    vertexai.init(project=project, location=location, credentials=get_credentials())

def should_continue(confidence_levels: List[float], max_tries: int) -> bool:
    return (not confidence_levels or confidence_levels[-1] < 0.9) and len(confidence_levels) < max_tries and not has_reached_plateau(confidence_levels)

def has_reached_plateau(values: List[float]) -> bool:
    if len(values) < 3:
        return False
    return values[-1] == values[-2] == values[-3]

#TODO: add tests for edge cases
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

    max_confidence_true = np.max(confidence_levels[shoplifting_detected_results]) if true_count else 0
    max_confidence_false = np.max(confidence_levels[~shoplifting_detected_results]) if false_count else 0

    return {
        'True Count': int(true_count),
        'False Count': int(false_count),
        'Average Confidence when True': round(float(avg_confidence_true), 4),
        'Average Confidence when False': round(float(avg_confidence_false), 4),
        'Highest Confidence when True': round(float(max_confidence_true), 4),
        'Highest Confidence when False': round(float(max_confidence_false), 4)
    }

def get_results_for_video(video_uri: str):
    init_vertex_ai(project="astral-sunbeam-443219-p5", location="us-central1")
    cv_model = ComputerVisionModel()
    prompt_model = PromptModel()
    analysis_model = AnalysisModel()
    confidence_levels = []
    shoplifting_detected_results = []
    prompt_model_responses = []
    cv_model_responses = []
    analysis_model_responses = []
    max_tries = 5

    while should_continue(confidence_levels, max_tries):
        video_file = Part.from_uri(uri=video_uri,
                                   mime_type="video/mp4")
        prompt_model_response = prompt_model.generate_prompt(video_file)
        cv_model_response = cv_model.analyze_video(video_file, prompt_model_response)
        analysis_model_response, shoplifting_detected, confidence_level = analysis_model.analyze_video_observations(
            video_file, cv_model_response)
        print(f"Shoplifting Detected: {shoplifting_detected}")
        print(f"Confidence Level: {confidence_level}")
        confidence_levels.append(confidence_level)
        shoplifting_detected_results.append(shoplifting_detected)
        prompt_model_responses.append(prompt_model_response)
        cv_model_responses.append(cv_model_response)
        analysis_model_responses.append(analysis_model_response)

    analysis = analyze_detection_results(confidence_levels, shoplifting_detected_results)
    print(analysis)

    result = {
        "video_uri": video_uri,
        "confidence_levels": confidence_levels,
        "shoplifting_detected_results": shoplifting_detected_results,
        "prompt_model_responses": prompt_model_responses,
        "cv_model_responses": cv_model_responses,
        "analysis_model_responses": analysis_model_responses,
        "analysis": analysis
    }

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pkl_path = f"video_analysis_{current_time}.pkl"

    with open(pkl_path, 'wb') as file:
        pickle.dump(result, file)

    print(f"Finished with {video_uri}, results saved to {pkl_path}")
    return result

def get_videos_uris_and_names(bucket_name):
    # Initialize the Google Cloud Storage client
    client = storage.Client(project="astral-sunbeam-443219-p5", credentials=get_credentials())
    # Get the bucket object
    bucket = client.get_bucket(bucket_name)
    # List all objects in the bucket and filter by .mp4 extension
    names_blobs = bucket.list_blobs()
    uris_blobs = bucket.list_blobs()
    names = [blob.name for blob in names_blobs if blob.name.endswith('.mp4') or blob.name.endswith('.MP4')]
    uris = [f"gs://{bucket_name}/{blob.name}" for blob in uris_blobs if blob.name.endswith('.mp4') or blob.name.endswith('.MP4')]

    return uris, names

#TODO: make the algorithm better. make it take into account also the False results.
#TODO: Make this into an AI model that gets results as number and output the prediction
def determine_shoplifting_likelihood(results: dict):
    # Calculate the weighted score
    count_weight = 0.5
    average_confidence_weight = 0.3
    highest_confidence_weight = 0.2

    # Normalize count to be between 0 and 1
    # Assuming the maximum expected count can be defined or observed from historical data
    # For demonstration, let's assume a maximum count of 10
    max_expected_count = 10
    normalized_count = results['True Count'] / (results["True Count"] + results["False Count"])
    normalized_count = min(normalized_count, 1)  # Ensure it does not exceed 1

    average_confidence = results['Average Confidence when True']
    highest_confidence = results['Highest Confidence when True']

    # Calculate the final score using the specified weights
    final_score = (count_weight * normalized_count) + \
                  (average_confidence_weight * average_confidence) + \
                  (highest_confidence_weight * highest_confidence)
    print(final_score)
    # Check if the final score exceeds the threshold
    return final_score

def load_pickle_files_from_directory(directory):
    # Dictionary to store the loaded data, keyed by file name
    loaded_data = {}

    # Walk through the directory
    for filename in os.listdir(directory):
        # Check if the file has a .pkl extension
        if filename.endswith('.pkl'):
            # Construct full file path
            file_path = os.path.join(directory, filename)
            # Open and load the pickle file
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                # Store the data with filename as key
                loaded_data[filename] = data

    return loaded_data

def main():
    uris, names = get_videos_uris_and_names("example_bucket_final_project")
    final_predictions = {}
    for uri, name in zip(uris, names):
        result = get_results_for_video(uri)
        final_predictions[name] = determine_shoplifting_likelihood(result["analysis"])

    with open("final_predictions.pkl", 'wb') as file:
        pickle.dump(final_predictions, file)

    print(final_predictions)
    print("Done")


if __name__ == "__main__":
    main()
    # all_videos_analysis = load_pickle_files_from_directory("/home/yonatan.r/PycharmProjects/SurveillanceAI/data-science/src/")
    # final_predictions = {}
    # for video_uri, video_analysis in all_videos_analysis.items():
    #     final_predictions[video_analysis["video_uri"]] = int(determine_shoplifting_likelihood(video_analysis["analysis"]) > 0.5)
    #
    # df = pd.DataFrame(list(final_predictions.values()), index=final_predictions.keys(), columns=['Prediction'])
    # # Export DataFrame to CSV, including the index
    # csv_file_path = 'output.csv'  # Specify your path and filename
    # df.to_csv(csv_file_path, index=True)  # Include index by setting index=True

