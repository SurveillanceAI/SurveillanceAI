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


class CvModel(GenerativeModel):

    default_system_instruction = [
        "You are a highly skilled security guard working at a retail store.",
        "Your main responsibility is to monitor customer behavior and identify potential shoplifting activities.",
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


    def analyze_video(self, video_file: Part, prompt: str) -> str:
        # Set contents to send to the model
        contents = [video_file, prompt]
        # Prompt the model to generate content
        response = self.generate_content(
            contents,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
        )

        return response.text


