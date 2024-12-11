from typing import Optional, List, Any
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import openai
from dotenv import load_dotenv
load_dotenv()
import os

class DummyLLM(LLM):
    def __init__(self):
        super().__init__()
        print("Dummy LLM loaded.")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        
        # Here, you can implement your own logic to generate a response
        # based on the given prompt
        print('Prompt:', prompt)
        response = f"This is a dummy response for the prompt: {prompt}"
        return response

    @property
    def _llm_type(self) -> str:
        return "DummyLLM"

from gradio_client import Client

class Qwen25LLM(LLM):
    def __init__(self, model_name: str = "Qwen/Qwen2.5"):
        super().__init__()
        self._client = Client(model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        result = self._client.predict(
		query = prompt,
		history=[],
		system="你是一个杰出的飞行器专家，懂得很多知识，可以有条理的回答相关问题",
		radio="72B",
		api_name="/model_chat"
        )
        response = result[1][0][1]['text']
        print(f"prompt is {prompt}")
        print(f"response is {response}")
        return f"{response}"

    @property
    def _llm_type(self) -> str:
        return "Qwen25LLM"