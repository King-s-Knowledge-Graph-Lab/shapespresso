import os

import instructor

from anthropic import Anthropic
from openai import OpenAI
from ollama import chat


class BaseModel:
    def __init__(self):
        pass


class GenerationModel(BaseModel):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def model_response(self, *args, **kwargs):
        raise NotImplementedError

    def structured_response(self, *args, **kwargs):
        raise NotImplementedError


class OllamaModel(GenerationModel):
    def __init__(self, model_name):
        super().__init__(model_name)

    def model_response(self, prompt):
        response = chat(
            self.model_name,
            messages=prompt,
            stream=False,
            options={
                "temperature": 0.1
            },
        )
        return response["message"]["content"]

    def structured_response(self, prompt, response_model, stream=False):
        response = chat(
            model=self.model_name,
            messages=prompt,
            format=response_model.model_json_schema(),
            stream=stream,
            options={
                "temperature": 0.1
            },
        )

        if stream:
            for chunk in response:
                print(chunk['message']['content'], end='', flush=True)

        return response_model.model_validate_json(response["message"]["content"])


class OpenAIModel(GenerationModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        if model_name == "deepseek-chat" or model_name == "deepseek-reasoner":
            self.client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=os.getenv("DEEPSEEK_API_KEY"),
            )
        elif "gpt" in model_name:
            self.client = OpenAI(
                base_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            raise NotImplementedError

    def model_response(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            stream=False,
            temperature=0.1,
        )
        if type(completion) is str:
            print("completion:", completion)
        response = completion.choices[0].message.content

        return response

    def structured_response(self, prompt, response_model):
        client = instructor.from_openai(
            self.client,
            mode=instructor.Mode.JSON,  # Instructor does not support multiple tool calls, use List[Model] instead
        )
        completion = client.chat.completions.create(
            model=self.model_name,
            response_model=response_model,
            messages=prompt,
            stream=False,
            temperature=0.1,
        )

        return completion

    def batch_request(self, batch_input_file_path):
        batch_input_file = self.client.files.create(
            file=open(batch_input_file_path, "rb"),
            purpose="batch",
        )
        batch_input_file_id = batch_input_file.id
        batch_metadata = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        return batch_metadata


class ClaudeModel(GenerationModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model_name = model_name
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def model_response(self, prompt):
        if prompt[0]["role"] == "system":
            system_content = prompt[0]["content"]
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.1,
                system=system_content,
                messages=prompt[1:],
            )
        else:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.1,
                messages=prompt,
            )

        return message.content[0].text
