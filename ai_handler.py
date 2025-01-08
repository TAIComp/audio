from openai import OpenAI
from typing import Iterator
from dataclasses import dataclass

@dataclass
class ChatConfig:
    """Configuration for chat completion"""
    model_name: str = "gpt-4"
    temperature: float = 0.7

@dataclass
class AnalysisConfig:
    """Configuration for sentence analysis"""
    model_name: str = "gpt-4"
    temperature: float = 0

class OpenAIService:
    def __init__(self):
        self.client = OpenAI()  # Will use OPENAI_API_KEY from environment

    def analyze_completion(self, text: str, config: AnalysisConfig) -> bool:
        """Analyze if the given text forms a complete thought/sentence."""
        try:
            response = self.client.chat.completions.create(
                model=config.model_name,
                messages=[
                    {"role": "system", "content": "You are a sentence completion analyzer. "
                     "Respond with ONLY 'true' or 'false' to indicate if the input forms a complete thought."},
                    {"role": "user", "content": text}
                ],
                temperature=config.temperature
            )
            return response.choices[0].message.content.strip().lower() == 'true'
        except Exception as e:
            print(f"Analysis error: {e}")
            return False

    def generate_chat_response(self, text: str, config: ChatConfig) -> Iterator[str]:
        """Generate streaming chat response with improved chunking."""
        try:
            response = self.client.chat.completions.create(
                model=config.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant engaged in verbal conversation. "
                     "Keep responses natural and concise. Use proper punctuation."},
                    {"role": "user", "content": text}
                ],
                temperature=config.temperature,
                stream=True
            )

            buffer = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    buffer += content
                    yield content
                    
        except Exception as e:
            yield f"Error: {str(e)}"

def create_default_service() -> OpenAIService:
    return OpenAIService()
