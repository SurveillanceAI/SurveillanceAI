from langchain_google_vertexai import VertexAI as LangChainVertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Image
)
from langgraph.graph import StateGraph, END
from typing import Dict, List, Optional, Union, TypedDict, Annotated
from vertexai.generative_models._generative_models import PartsType, GenerationConfigType, SafetySettingsType


class VideoAnalysisState(TypedDict):
    video_file: Part
    video_description: Optional[str]
    analysis_result: Optional[str]
    system_instruction: str


class PromptModel:
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

        # Initialize LangChain VertexAI
        self.langchain_llm = LangChainVertexAI(
            model_name=model_name,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            max_output_tokens=generation_config.max_output_tokens,
        )

        # Create a prompt template for video description
        self.video_description_template = PromptTemplate(
            input_variables=["video"],
            template="""
            Please provide a detailed description of the video, focusing on:
            1. Customer movements and behaviors
            2. Interactions with store items and environment
            3. Temporal sequence of events
            4. Notable actions or patterns
            5. Relevant details about the store layout and context
            
            Video: {video}
            """
        )

        # Create a prompt template for analysis
        self.video_analysis_template = PromptTemplate(
            input_variables=["system_instruction", "video_description"],
            template="""
            {system_instruction}
            
            Based on the following video description, generate a detailed prompt that will help the CV model identify potential shoplifting activities. Focus specifically on:
            1. Suspicious movements and behaviors that might indicate shoplifting
            2. Patterns of customer interaction with merchandise
            3. Any attempts to conceal items or avoid detection
            4. Customer paths through the store, especially near exits
            5. Interactions (or lack thereof) with store staff and payment areas
            
            Video Description:
            {video_description}
            
            Generate a structured analysis following the chain-of-thought method, breaking down the complex task of shoplifting detection into clear, actionable steps.
            """
        )

        # Create the LangChains
        self.description_chain = LLMChain(
            llm=self.langchain_llm,
            prompt=self.video_description_template
        )
        
        self.analysis_chain = LLMChain(
            llm=self.langchain_llm,
            prompt=self.video_analysis_template
        )

        # Create the workflow graph
        self.workflow = self._create_workflow()

    def _describe_video(self, state: Annotated[VideoAnalysisState, "The current state"]) -> VideoAnalysisState:
        """Generate a description of the video"""
        video_description = self.description_chain.run(video=str(state["video_file"]))
        state["video_description"] = video_description
        return state

    def _analyze_video(self, state: Annotated[VideoAnalysisState, "The current state"]) -> VideoAnalysisState:
        """Analyze the video description for shoplifting behavior"""
        result = self.analysis_chain.run(
            system_instruction=state["system_instruction"],
            video_description=state["video_description"]
        )
        state["analysis_result"] = result
        return state

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        # Create the graph
        workflow = StateGraph(VideoAnalysisState)

        # Add nodes for each step
        workflow.add_node("describe_video", self._describe_video)
        workflow.add_node("analyze_video", self._analyze_video)

        # Create the edges
        workflow.add_edge("describe_video", "analyze_video")
        workflow.add_edge("analyze_video", END)

        # Set the entry point
        workflow.set_entry_point("describe_video")

        return workflow.compile()

    def analyze_video_for_shoplifting(self, video_file: Part) -> str:
        """Run the video analysis workflow"""
        # Initialize the state
        initial_state = {
            "video_file": video_file,
            "video_description": None,
            "analysis_result": None,
            "system_instruction": "\n".join(self.default_system_instruction)
        }

        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state["analysis_result"]


