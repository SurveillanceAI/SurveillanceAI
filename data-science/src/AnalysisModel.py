from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Image
)

from typing import Dict, List, Optional, Union
from vertexai.generative_models._generative_models import PartsType, GenerationConfigType, SafetySettingsType
from typing import Tuple
import re

class AnalysisModel(GenerativeModel):

    default_system_instruction = [
        "As an advanced analytical model, your main function is to process the observations provided by the cv_model and the corresponding video footage to determine the likelihood of shoplifting activity.",
        "Your role involves analyzing the video and the detailed observations to validate the presence of shoplifting behaviors and calculate a confidence score between 0 and 1.",
        "You must provide two outputs: a Boolean value (True if shoplifting is detected, False otherwise) and a numerical confidence score indicating the certainty of your analysis.",
        "Utilize advanced video analysis techniques such as behavior pattern recognition, object detection, and movement tracking to reassess the scenes described by the cv_model.",
        "Your output should clearly state whether shoplifting is likely or not, accompanied by a confidence score that quantifies the certainty of this determination.",
        "Ensure your analysis is robust, fair, and free from bias, focusing strictly on the actions depicted in the video."
    ]

    default_generation_config = GenerationConfig(
        temperature=0.9,
        top_p=1.0,
        top_k=32,
        candidate_count=1,
        max_output_tokens=8192,
    )

    # Set safety settings.
    # Dont know what does it mean, need to investigate
    default_safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    def __init__(self,
        model_name: str = "gemini-1.5-flash-002",
        *,
        generation_config: Optional[GenerationConfigType] = None,
        safety_settings: Optional[SafetySettingsType] = None,
        system_instruction: Optional[PartsType] = None,
        labels: Optional[Dict[str, str]] = None):

        if system_instruction is None:
            # system_instruction = Explation to the model to exaplin who he is, before it is given a specific task.
            system_instruction = self.default_system_instruction

        if generation_config is None:
            generation_config = self.default_generation_config

        if safety_settings is None:
            safety_settings = self.default_safety_settings

        super().__init__(model_name=model_name
                         , generation_config=generation_config
                         , safety_settings=safety_settings
                         , system_instruction=system_instruction
                         , labels=labels)


    def analyze_video_observations(self, video_file: Part, video_observations: str, prompt: str = "Analyze the provided video and cv_model's observations to assess the likelihood of shoplifting. "
                                                                                                  "Determine if shoplifting has occurred and provide outputs labeled 'Shoplifting Detected' for the Boolean determination "
                                                                                                  "and 'Confidence Level' for the confidence score between 0 and 1.") -> Tuple[str, bool, float]:
        prompt += "The cv_model response here is: " + video_observations
        contents = [video_file, prompt]
        # Prompt the model to generate content
        analysis_model_response = self.generate_content(
            contents,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
        )

        shoplifting_detected, confidence_level = self._extract_values_from_response(analysis_model_response.text)
        return analysis_model_response.text, shoplifting_detected, confidence_level

    def _extract_values_from_response(self, response_str: str) -> Tuple[bool, float]:
        response_str = response_str.lower()
        # Finding the start index of 'Shoplifting Detected' and extracting the Boolean value
        start_detected = response_str.find("shoplifting detected")
        if start_detected != -1:  # Check if the substring was found
            true_index = response_str.find("true", start_detected, start_detected + len("shoplifting detected") + 10)
            false_index = response_str.find("false", start_detected, start_detected + len("shoplifting detected") + 10)

            # Convert the extracted value to Boolean
            if true_index != -1:
                shoplifting_detected = True
            else:
                shoplifting_detected = False
        else:
            shoplifting_detected = False

        # Finding the start index of 'Confidence Level' and extracting the float value
        start_confidence = response_str.find("confidence level")
        if start_confidence != -1:  # Check if the substring was found
            end_confidence = start_confidence + len("confidence level") + 10
            confidence_level_str = response_str[start_confidence:end_confidence].strip()
            confidence_level = self._extract_float(confidence_level_str)
        else:
            confidence_level = 0.0

        return shoplifting_detected, confidence_level

    def _extract_float(self, s: str) -> float:
        match = re.search(r'\d+(\.\d+)?', s)
        if match:
            return float(match.group())
        else:
            return 0.0


