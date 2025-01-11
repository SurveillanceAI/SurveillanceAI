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


class PromptModel(GenerativeModel):

    default_system_instruction = [
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


    def generate_prompt(self, video_file: Part, prompt: str = "Generate a prompt that instructs the cv_model to effectively monitor customer behavior"
                                                              " and identify potential shoplifting activities in a retail environment in the attached video,"
                                                              "following the 'chain of thought' method to break down complex tasks into smaller, more manageable steps.") -> str:
        # Set contents to send to the model
        contents = [video_file, prompt]
        # Prompt the model to generate content
        response = self.generate_content(
            contents,
            generation_config=self._generation_config,
            safety_settings=self._safety_settings,
        )

        return response.text


