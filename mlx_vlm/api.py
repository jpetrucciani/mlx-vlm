from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import argparse
import base64
import io
import os
import time
import uvicorn

from .prompt_utils import get_chat_template, get_message_json
from .utils import load, load_config, stream_generate


class ImageContent(BaseModel):
    image_url: str = Field(..., description="Base64-encoded image data")


class Message(BaseModel):
    role: str = Field(
        ..., description="Role of the message sender (system/user/assistant)"
    )
    content: Union[str, List[Union[str, ImageContent]]] = Field(
        ..., description="Message content"
    )


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model to use for completion")
    messages: List[Message] = Field(..., description="Messages in the conversation")
    temperature: Optional[float] = Field(0.1, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        200, description="Maximum number of tokens to generate"
    )
    stream: Optional[bool] = Field(False, description="Whether to stream the response")


class DeltaContent(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Optional[Message] = None
    delta: Optional[DeltaContent] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = "chat.completion"
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatCompletionResponseChoice]


def create_app(model_path: str):
    app = FastAPI(title="MLX-VLM API")

    # Load model and config
    config = load_config(model_path)
    model, processor = load(model_path, kwargs={"trust_remote_code": True})

    def process_base64_image(base64_string: str) -> Image.Image:
        try:
            # Handle both "data:image/..." and plain base64 formats
            if ";base64," in base64_string:
                base64_string = base64_string.split(";base64,")[1]
            elif "base64," in base64_string:
                base64_string = base64_string.split("base64,")[1]

            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if needed
            if image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                background = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode in ("RGBA", "LA"):
                    background.paste(image, mask=image.getchannel("A"))
                else:
                    background.paste(image)
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            return image
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

    def extract_image_and_text(
        message_content: Union[str, List[Union[str, dict]]]
    ) -> tuple:
        """Extract image and text from message content."""
        print("Received message content type:", type(message_content))
        if isinstance(message_content, str):
            print("Content is string:", message_content[:100])
            return None, message_content

        text_parts = []
        image = None

        print("Processing content parts...")
        for content in message_content:
            print("Content part type:", type(content))
            if isinstance(content, str):
                print("Found text part:", content)
                text_parts.append(content)
            elif hasattr(content, "image_url") or isinstance(content, dict):
                print("Found image content")
                # Handle both ImageContent object and dict
                image_data = (
                    content.image_url
                    if hasattr(content, "image_url")
                    else content.get("image_url", "")
                )
                print("Image data type:", type(image_data))
                if isinstance(image_data, dict):
                    print("Nested image_url object with keys:", image_data.keys())
                    image_data = image_data.get("url", "")

                if isinstance(image_data, str) and (
                    image_data.startswith("data:image/")
                    or image_data.startswith("base64,")
                ):
                    try:
                        image = process_base64_image(image_data)
                        if image is None:
                            raise HTTPException(
                                status_code=400, detail="Failed to process image data"
                            )
                    except Exception as e:
                        raise HTTPException(
                            status_code=400, detail=f"Failed to process image: {str(e)}"
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Only base64 encoded images are supported",
                    )

        return image, " ".join(text_parts)

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest):
        print("Received request with messages:", len(request.messages))
        if config["model_type"] != "paligemma":
            # Process messages
            chat_history = []
            latest_image = None

            for msg in request.messages:
                print(f"Processing message with role: {msg.role}")
                print(f"Message content type: {type(msg.content)}")
                image, text = extract_image_and_text(msg.content)
                print(f"Extracted image: {image is not None}, text: {text}")
                if image is not None:
                    latest_image = image
                chat_history.append({"role": msg.role, "content": text})

            if latest_image is None:
                raise HTTPException(
                    status_code=400,
                    detail="An image is required for chat completion. Ensure the image_url contains valid base64-encoded image data.",
                )

            # Format messages according to model requirements
            messages = []
            for i, m in enumerate(chat_history):
                skip_token = True
                if i == len(chat_history) - 1 and m["role"] == "user":
                    skip_token = False
                messages.append(
                    get_message_json(
                        config["model_type"],
                        m["content"],
                        role=m["role"],
                        skip_image_token=skip_token,
                    )
                )

            messages = get_chat_template(
                processor, messages, tokenize=False, add_generation_prompt=True
            )
        else:
            raise HTTPException(
                status_code=400, detail="Paligemma model type not supported"
            )

        # Generate response
        response_text = ""
        for chunk in stream_generate(
            model,
            processor,
            messages,
            latest_image,
            temp=request.temperature,
            max_tokens=request.max_tokens,
        ):
            response_text += chunk.text

        return ChatCompletionResponse(
            id="chatcmpl-" + base64.b32encode(os.urandom(5)).decode("utf-8").lower(),
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
        )

    return app


def parse_arguments():
    parser = argparse.ArgumentParser(description="Start MLX-VLM API server")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/SmolVLM-Instruct-8bit",
        help="The path to the local model directory or Hugging Face repo",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="The port to run the API server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="The host address to run the API server on",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    app = create_app(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
