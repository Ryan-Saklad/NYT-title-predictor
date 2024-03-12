import os
from collections.abc import Iterator
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import app.config
from app.models.language_model import LanguageModel, HyperparametersBase

SUPPORTED_MODELS: list[str] = ["gemini-pro"]

class GeminiHyperparameters(HyperparametersBase):
    def __init__(
        self,
        candidate_count: int | None = None,
        stop_sequences: list[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        self.candidate_count = candidate_count
        self.stop_sequences = stop_sequences or []
        self.max_output_tokens = max_output_tokens # TODO: This currently doesn't work
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
    def to_dict(self) -> dict[str, any]:
        return {
            k: v
            for k, v in self.__dict__.items()
            if v is not None
        }

class Gemini(LanguageModel):
    """Gemini model interface."""
    
    DEFAULT_SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    def __init__(
        self,
        model_name: str = "gemini-pro",
        api_key: str | None = None,
        default_hyperparameters: GeminiHyperparameters | None = None
    ) -> None:
        """Initialize the Gemini model.
        
        Args:
            model_name: The name of the model to use.
            version: The model version to use.
            api_key: The API key for accessing the Gemini API. If not provided, it will be loaded from the environment variable GEMINI_API_KEY.
            default_hyperparameters: Default hyperparameters to use for generation.
        """

        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {', '.join(SUPPORTED_MODELS)}")
        
        api_key: str = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found. Please provide it or set the GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        
        self.default_hyperparameters = default_hyperparameters or GeminiHyperparameters()
        self.model_name = model_name
        
    def generate(
        self,
        prompt: str, 
        history: list[dict[str, str]] | None = None,
        hyperparameters: GeminiHyperparameters | None = None,
        stream: bool = False,
    ) -> Iterator[str] | str:
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
            
        hp = hyperparameters or GeminiHyperparameters()
        
        # Only valid roles are: user, model
        if history and any(h["role"] == "system" for h in history):
            raise ValueError("Role 'system' is not supported.")

        messages: list[dict[str, list[dict[str, str]]]] = [{"role": "model" if h["role"] == "assistant" else h["role"], "parts": [{"text": h["content"]}] } for h in history] if history else []
        messages.append({"role": "user", "parts": [{"text": prompt}]})
        
        if hp.candidate_count is not None and not 1 <= hp.candidate_count <= 8:
            # Currently a maximum of one candidate is allowed
            raise ValueError("Candidate count must be between 1 and 8.")
        if hp.temperature is not None and not 0.0 <= hp.temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0.")
        if hp.top_p is not None and not 0.0 <= hp.top_p <= 1.0:
            raise ValueError("TopP must be between 0.0 and 1.0.")
            
        model_info = genai.get_model(name=f"models/{self.model_name}")
        max_output_tokens: int = model_info.output_token_limit
        if hp.max_output_tokens is not None and hp.max_output_tokens > max_output_tokens:
            raise ValueError(f"Max output tokens must not exceed {max_output_tokens}.")

        if hp.max_output_tokens is not None:
            raise NotImplementedError("Stopping generation due to max_output_tokens without returning a response is not currently supported.")

        generation_config = genai.types.GenerationConfig(
            **hp.to_dict()
        )
        
        response = self.model.generate_content(
            messages,
            stream=stream,
            generation_config=generation_config,
            safety_settings=self.DEFAULT_SAFETY_SETTINGS,
        )

        if stream:
            return (chunk.text for chunk in response)
        else:
            return response.text
