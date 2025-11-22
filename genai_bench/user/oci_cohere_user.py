from locust import task

import json
import time
from typing import Any, Callable, List, Optional

from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    ChatResult,
    CohereChatRequest,
    DedicatedServingMode,
    EmbedTextDetails,
    EmbedTextResult,
    GenericChatRequest,
    OnDemandServingMode,
    RerankTextDetails,
)

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserImageEmbeddingRequest,
    UserRequest,
    UserReRankRequest,
    UserResponse,
)
from genai_bench.user.base_user import BaseUser

logger = init_logger(__name__)

# Monkey-patch the OCI SDK to allow "COHEREV2" as a valid api_format value
# The SDK only validates "COHERE" or "GENERIC", but the API accepts "COHEREV2"
from oci.generative_ai_inference.models.base_chat_request import BaseChatRequest
from oci.util import value_allowed_none_or_none_sentinel

# Get the original getter
_original_api_format_getter = BaseChatRequest.api_format.fget

# Create a new setter that allows COHEREV2
def _new_api_format_setter(self, api_format):
    allowed_values = ["COHERE", "GENERIC", "COHEREV2"]
    if not value_allowed_none_or_none_sentinel(api_format, allowed_values):
        raise ValueError(
            f"Invalid value for `api_format`, must be None or one of {allowed_values}"
        )
    self._api_format = api_format

# Replace the property with the original getter and new setter
BaseChatRequest.api_format = property(_original_api_format_getter, _new_api_format_setter)


class OCICohereUser(BaseUser):
    """User class for Cohere model API with OCI authentication."""

    BACKEND_NAME = "oci-cohere"
    supported_tasks = {
        "text-to-text": "chat",
        "image-text-to-text": "chat",
        "text-to-rerank": "rerank",
        "text-to-embeddings": "embeddings",
        "image-to-embeddings": "embeddings",
    }
    host: Optional[str] = None
    auth_provider: Optional[ModelAuthProvider] = None

    def on_start(self):
        """Initialize OCI client on start."""
        super().on_start()
        if not self.auth_provider:
            raise ValueError("Auth is required for OCICohereUser")
        self.client = GenerativeAiInferenceClient(
            config=self.auth_provider.get_config(),
            signer=self.auth_provider.get_credentials(),
            service_endpoint=self.host,
        )
        logger.debug("Generative AI Inference Client initialized.")

    def _truncate_image_urls_for_logging(self, payload: Any) -> Any:
        """Truncate base64 image URLs in payload for logging to avoid huge log entries.
        
        Args:
            payload: The payload object (ChatDetails) to process
            
        Returns:
            A dict representation with truncated image URLs
        """
        def truncate_url(url: str) -> str:
            """Truncate base64 image URL to first 5 chars of base64 data."""
            if url.startswith("data:image/"):
                # Extract base64 part from data URL
                parts = url.split(",", 1)
                if len(parts) == 2:
                    prefix = parts[0]  # "data:image/jpeg;base64"
                    base64_data = parts[1]
                    # Truncate base64 data to first 5 chars
                    truncated = base64_data[:5] if len(base64_data) > 5 else base64_data
                    return f"{prefix},{truncated}..."
                return url
            elif url.startswith(("http://", "https://")):
                # Leave HTTP URLs as-is
                return url
            else:
                # Assume it's base64, truncate to first 5 chars
                if len(url) > 5:
                    return f"{url[:5]}..."
                return url
        
        def process_messages(messages: List[dict]) -> List[dict]:
            """Process messages array to truncate image URLs."""
            processed = []
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        processed_content = []
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "IMAGE_URL" and "imageUrl" in item:
                                    image_url_obj = item["imageUrl"]
                                    if isinstance(image_url_obj, dict) and "url" in image_url_obj:
                                        # Create a copy and truncate the URL
                                        processed_item = item.copy()
                                        processed_item["imageUrl"] = image_url_obj.copy()
                                        processed_item["imageUrl"]["url"] = truncate_url(image_url_obj["url"])
                                        processed_content.append(processed_item)
                                    else:
                                        processed_content.append(item)
                                else:
                                    processed_content.append(item)
                            else:
                                processed_content.append(item)
                        processed_msg = msg.copy()
                        processed_msg["content"] = processed_content
                        processed.append(processed_msg)
                    else:
                        processed.append(msg)
                else:
                    processed.append(msg)
            return processed
        
        # Convert payload to dict for processing
        try:
            # Access payload attributes directly
            payload_dict = {}
            
            # Extract compartment_id
            if hasattr(payload, "compartment_id"):
                payload_dict["compartment_id"] = payload.compartment_id
            
            # Extract serving_mode
            if hasattr(payload, "serving_mode"):
                serving_mode = payload.serving_mode
                if hasattr(serving_mode, "__dict__"):
                    serving_mode_dict = {}
                    if hasattr(serving_mode, "serving_type"):
                        serving_mode_dict["serving_type"] = serving_mode.serving_type
                    if hasattr(serving_mode, "model_id"):
                        serving_mode_dict["model_id"] = serving_mode.model_id
                    if hasattr(serving_mode, "endpoint_id"):
                        serving_mode_dict["endpoint_id"] = serving_mode.endpoint_id
                    payload_dict["serving_mode"] = serving_mode_dict
                else:
                    payload_dict["serving_mode"] = str(serving_mode)
            
            # Extract and process chat_request
            if hasattr(payload, "chat_request"):
                chat_request = payload.chat_request
                chat_request_dict = {}
                
                # Extract api_format
                if hasattr(chat_request, "api_format"):
                    chat_request_dict["api_format"] = chat_request.api_format
                
                # Extract and process messages
                if hasattr(chat_request, "messages"):
                    messages = chat_request.messages
                    if messages:
                        processed_messages = process_messages(messages)
                        chat_request_dict["messages"] = processed_messages
                
                # Extract other common fields
                for attr in ["max_tokens", "temperature", "top_p", "top_k", 
                            "frequency_penalty", "presence_penalty", "is_stream"]:
                    if hasattr(chat_request, attr):
                        chat_request_dict[attr] = getattr(chat_request, attr)
                
                payload_dict["chat_request"] = chat_request_dict
            
            return payload_dict
        except Exception as e:
            logger.debug(f"Error truncating image URLs for logging: {e}, using original payload")
            return payload
    
    def send_request(
        self,
        make_request: Callable,
        endpoint: str,
        payload: Any,
        parse_strategy: Any,
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        # Truncate image URLs in payload for logging
        payload_for_logging = self._truncate_image_urls_for_logging(payload)
        logger.debug(f"Sending request with payload: {payload_for_logging}")
        response = None
        try:
            start_time = time.monotonic()
            response = make_request()
            non_stream_post_end_time = time.monotonic()

            if response.status == 200:
                metrics_response = parse_strategy(
                    payload,
                    response,
                    start_time,
                    num_prefill_tokens,
                    non_stream_post_end_time,
                )
            else:
                request_id = getattr(response, "request_id", "N/A")
                logger.warning(
                    f"Received error status-code: {response.status} "
                    f"RequestId: {request_id}, "
                    f"Response: {response.response}"
                )
                metrics_response = UserResponse(
                    status_code=response.status,
                    error_message="Request Failed",
                )
            self.collect_metrics(metrics_response, endpoint)
            return metrics_response
        except Exception as e:
            # Enhanced error logging with request details
            error_type = type(e).__name__
            error_message = str(e)
            
            # Try to extract request_id from response if available
            request_id = "N/A"
            if response is not None:
                request_id = getattr(response, "request_id", "N/A")
            else:
                # Try to extract from exception context
                if hasattr(e, "__cause__") and hasattr(e.__cause__, "response"):
                    nested_response = getattr(e.__cause__, "response", None)
                    if nested_response and hasattr(nested_response, "request_id"):
                        request_id = nested_response.request_id
                elif hasattr(e, "response") and hasattr(e.response, "request_id"):
                    request_id = e.response.request_id
            
            # Log payload info (truncated) for debugging
            payload_info = "N/A"
            try:
                if hasattr(payload, "chat_request"):
                    chat_req = payload.chat_request
                    if hasattr(chat_req, "api_format"):
                        payload_info = f"api_format={chat_req.api_format}"
                    if hasattr(chat_req, "messages") and chat_req.messages:
                        msg_count = len(chat_req.messages)
                        payload_info += f", messages_count={msg_count}"
            except Exception:
                pass
            
            logger.error(
                f"Error sending request to {endpoint} - "
                f"RequestId: {request_id}, "
                f"ErrorType: {error_type}, "
                f"ErrorMessage: {error_message}, "
                f"PayloadInfo: {payload_info}"
            )
            
            error_response = UserResponse(
                status_code=500,
                error_message=str(e),
                num_prefill_tokens=num_prefill_tokens or 0,
            )
            self.collect_metrics(error_response, endpoint)
            return error_response

    @task
    def chat(self):
        """Send a chat completion request using Cohere format."""
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"Expected UserChatRequest for OCICohereUser.chat, got "
                f"{type(user_request)}"
            )

        compartment_id = self.get_compartment_id(user_request)
        serving_mode = self.get_serving_mode(user_request)

        # Handle vision requests (image-text-to-text)
        images = None
        if isinstance(user_request, UserImageChatRequest):
            images = user_request.image_content

        # COHEREV2 format uses messages array structure (like OpenAI), not message field
        # Build messages array according to COHEREV2 format
        messages = []
        
        # Add chat history if provided (COHEREV2 format uses messages array for history)
        if "chatHistory" in user_request.additional_request_params:
            chat_history = user_request.additional_request_params["chatHistory"]
            for msg in chat_history:
                # Convert chat history messages to COHEREV2 format
                role = msg.get("role", "user").upper()
                if role not in ["USER", "ASSISTANT", "SYSTEM"]:
                    role = "USER"  # Default to USER if unknown
                
                # Handle content - could be string or array
                if isinstance(msg.get("content"), str):
                    content = [{"type": "TEXT", "text": msg["content"]}]
                else:
                    # Assume it's already in the right format or convert
                    content = msg.get("content", [])
                
                messages.append({"role": role, "content": content})
        
        # Build current user message content
        content = [{"type": "TEXT", "text": user_request.prompt}]
        
        # Add images if this is a vision request
        # COHEREV2 format: images are in content array with type "IMAGE_URL" or base64
        if images is not None:
            for image in images:
                if image.startswith("data:image/"):
                    # Use full data URL for base64 images
                    content.append({
                        "type": "IMAGE_URL",
                        "imageUrl": {
                            "url": image  # Use full data URL
                        }
                    })
                elif image.startswith(("http://", "https://")):
                    # HTTP URL
                    content.append({
                        "type": "IMAGE_URL",
                        "imageUrl": {
                            "url": image,
                            "detail": "AUTO"
                        }
                    })
                else:
                    # Assume base64 - convert to data URL format
                    content.append({
                        "type": "IMAGE_URL",
                        "imageUrl": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    })
        
        messages.append({"role": "USER", "content": content})
        
        # Note: documents are not directly supported in COHEREV2 messages format
        # They would need to be included in the message content if needed
        
        # Construct chat request using GenericChatRequest for COHEREV2 format
        # COHEREV2 requires messages array, not message field
        chat_request = GenericChatRequest(
            api_format="COHEREV2",
            messages=messages,
            max_tokens=user_request.max_tokens,
            is_stream=True,
            temperature=user_request.additional_request_params.get("temperature", 0.1),
            top_p=user_request.additional_request_params.get("topP", 0),
            top_k=user_request.additional_request_params.get("topK", 0.75),
            frequency_penalty=user_request.additional_request_params.get(
                "frequencyPenalty", 0
            ),
            presence_penalty=user_request.additional_request_params.get(
                "presencePenalty", 0
            ),
        )

        # Define payload with compartment ID and serving mode
        chat_detail = ChatDetails(
            compartment_id=compartment_id,
            serving_mode=serving_mode,
            chat_request=chat_request,
        )
        return self.send_request(
            make_request=lambda: self.client.chat(chat_detail),
            endpoint="chat",
            payload=chat_detail,
            parse_strategy=self.parse_chat_response,
            num_prefill_tokens=user_request.num_prefill_tokens,
        )

    @task
    def embeddings(self):
        """Send an embedding request using Cohere format."""
        user_request = self.sample()

        if not isinstance(user_request, UserEmbeddingRequest):
            raise AttributeError(
                f"user_request should be of type UserEmbeddingRequest for "
                f"OCICohereUser.embeddings, got {type(user_request)}"
            )

        if user_request.documents and not user_request.num_prefill_tokens:
            logger.warning(
                "Number of prefill tokens is missing or 0. Please double check."
            )

        # Retrieve compartment ID and serving mode
        compartment_id = self.get_compartment_id(user_request)
        serving_mode = self.get_serving_mode(user_request)
        input_type = self.get_embedding_input_type(user_request)
        inputs = self.get_inputs(user_request)

        embed_text_detail = EmbedTextDetails(
            compartment_id=compartment_id,
            serving_mode=serving_mode,
            inputs=inputs,
            input_type=input_type,
            truncate=user_request.additional_request_params.get("truncate", "NONE"),
        )

        response = self.send_request(
            make_request=lambda: self.client.embed_text(embed_text_detail),
            endpoint="embedText",
            payload=embed_text_detail,
            parse_strategy=self.parse_embedding_response,
            num_prefill_tokens=user_request.num_prefill_tokens,
        )

        logger.debug(f"Response received {response}")

    @task
    def rerank(self):
        """Send an rerank request using Cohere format."""
        user_request = self.sample()

        if not isinstance(user_request, UserReRankRequest):
            raise AttributeError(
                f"user_request should be of type UserReRankRequest for "
                f"OCICohereUser.rerank, got {type(user_request)}"
            )

        if user_request.documents and not user_request.num_prefill_tokens:
            logger.error(
                "Number of prefill tokens is missing or 0. Please double check."
            )

        # Retrieve compartment ID and serving mode
        compartment_id = self.get_compartment_id(user_request)
        top_n = self.get_top_n(user_request)
        serving_mode = self.get_serving_mode(user_request)
        documents = self.get_documents(user_request)
        query = self.get_query(user_request)

        # TODO: Re-rank3.5 API Changes for OCI GenAI are in progress
        rerank_text_details = RerankTextDetails(
            compartment_id=compartment_id,
            serving_mode=serving_mode,
            documents=documents,
            input=query,
            top_n=top_n,
        )

        response = self.send_request(
            make_request=lambda: self.client.rerank_text(rerank_text_details),
            endpoint="rerankText",
            payload=rerank_text_details,
            parse_strategy=self.parse_rerank_response,
            num_prefill_tokens=user_request.num_prefill_tokens,
        )

        logger.debug(f"Response received {response}")

    def get_compartment_id(self, user_request: UserRequest):
        compartment_id = user_request.additional_request_params.get("compartmentId")
        if not compartment_id:
            raise ValueError("compartmentId missing in additional request params")
        return compartment_id

    def get_top_n(self, user_request: UserRequest):
        return user_request.additional_request_params.get("topN")

    def get_embedding_input_type(self, user_request) -> str:
        input_type = "SEARCH_DOCUMENT"
        if isinstance(user_request, UserImageEmbeddingRequest):
            input_type = "IMAGE"
        return input_type

    def get_documents(self, re_rank_request: UserReRankRequest) -> List[Any]:
        return re_rank_request.documents

    def get_query(self, re_rank_request: UserReRankRequest) -> str:
        return re_rank_request.query

    def get_inputs(self, user_request) -> List[Any]:
        if isinstance(user_request, UserImageEmbeddingRequest):
            num_sampled_images = len(user_request.image_content)
            if num_sampled_images > 1:
                raise ValueError(
                    f"OCI-Cohere Image embedding supports only 1 "
                    f"image but, the value provided in traffic"
                    f"scenario is requesting {num_sampled_images}"
                )
            return user_request.image_content
        return user_request.documents

    def get_serving_mode(self, user_request: UserRequest) -> Any:
        params = user_request.additional_request_params
        model_id = user_request.model
        serving_type = params.get("servingType", "ON_DEMAND")
        if serving_type == "DEDICATED":
            endpoint_id = params.get("endpointId")
            if not endpoint_id:
                raise ValueError(
                    "endpointId must be provided for DEDICATED servingType"
                )
            logger.debug(
                f"Using DedicatedServingMode {serving_type} with "
                f"endpoint ID: {endpoint_id}"
            )
            return DedicatedServingMode(endpoint_id=endpoint_id)
        else:
            logger.debug(
                f"Using OnDemandServingMode {serving_type} with model ID: {model_id}"
            )
            return OnDemandServingMode(model_id=model_id)

    def parse_chat_response(
        self,
        request: ChatDetails,
        response: ChatResult,
        start_time: float,
        num_prefill_tokens: Optional[int],
        _: float,
    ) -> UserResponse:
        """
        Parses the streaming response from the Cohere API in OCI format.

        Args:
            request (ChatDetails): OCICohere Chat request.
            response (ChatResult): The streaming response from the Cohere API.
            start_time (float): Timestamp of request initiation.
            num_prefill_tokens (Optional[int]): Number of tokens in the prompt.
                For vision requests, this may be None and will be estimated from the text prompt.
            _ (float): Placeholder for unused variable.

        Returns:
            UserResponse: Parsed response in the UserResponse format.
        """
        generated_text = ""
        tokens_received = 0
        time_at_first_token: Optional[float] = None
        finish_reason = None
        previous_data = None

        # Iterate over each event in the streaming response
        try:
            for event in response.data.events():
                # Raw event data from the stream
                event_data = event.data.strip()

                # Parse the event data as JSON
                try:
                    parsed_data = json.loads(event_data)
                    finish_reason = parsed_data.get("finishReason", None)
                    if not finish_reason:
                        # Extract text content if present
                        # COHEREV2 format might have different structure, check multiple possible fields
                        text_segment = parsed_data.get("text", "")
                        if not text_segment:
                            # Try alternative fields for COHEREV2 format
                            if "message" in parsed_data:
                                message = parsed_data.get("message", {})
                                if isinstance(message, dict):
                                    content = message.get("content", [])
                                    if isinstance(content, list) and len(content) > 0:
                                        # COHEREV2 format supports multiple content types:
                                        # - TEXT: has "text" field
                                        # - THINKING: has "thinking" field (for reasoning models)
                                        # - Other types may exist
                                        for content_item in content:
                                            if isinstance(content_item, dict):
                                                content_type = content_item.get("type", "")
                                                if content_type == "TEXT":
                                                    text_segment = content_item.get("text", "")
                                                    if text_segment:
                                                        break
                                                elif content_type == "THINKING":
                                                    # For reasoning models, include thinking content
                                                    thinking_text = content_item.get("thinking", "")
                                                    if thinking_text:
                                                        text_segment = thinking_text
                                                        break
                                                elif "text" in content_item:
                                                    # Fallback: check for text field in any content type
                                                    text_segment = content_item.get("text", "")
                                                    if text_segment:
                                                        break
                                elif isinstance(message, str):
                                    text_segment = message
                        
                        if text_segment:
                            # Capture the time at the first token
                            if not time_at_first_token:
                                time_at_first_token = time.monotonic()
                            generated_text += text_segment
                            tokens_received += 1  # each event contains one token
                            logger.debug(f"number of tokens received: {tokens_received}")
                        else:
                            # Log the structure if no text found (for debugging COHEREV2 format)
                            logger.debug(f"No text found in parsed_data: {parsed_data}")
                        # Track the previous data for debugging purposes
                        previous_data = parsed_data
                    else:
                        # we have reached the end
                        logger.debug(
                            f"We have reached the end of the response "
                            f"with finish reason: {finish_reason}"
                        )
                        break
                except json.JSONDecodeError:
                    logger.warning(
                        f"Error decoding JSON from event data: {event_data}, "
                        f"previous data: {previous_data}, "
                        f"finish reason: {finish_reason}"
                    )
                    continue
        except Exception as stream_error:
            # Enhanced error logging with request details
            request_id = getattr(response, "request_id", "N/A")
            response_status = getattr(response, "status", "N/A")
            error_type = type(stream_error).__name__
            error_message = str(stream_error)
            
            logger.error(
                f"Error reading streaming response - "
                f"RequestId: {request_id}, "
                f"Status: {response_status}, "
                f"ErrorType: {error_type}, "
                f"ErrorMessage: {error_message}, "
                f"TokensReceived: {tokens_received}, "
                f"GeneratedTextLength: {len(generated_text)}"
            )
            # Re-raise to be handled by send_request
            raise

        # End timing for response
        end_time = time.monotonic()
        
        # Ensure time_at_first_token is never None (fallback to end_time)
        # This can happen if no text tokens are received in the stream
        if time_at_first_token is None:
            time_at_first_token = end_time
            logger.warning(
                "time_at_first_token was None, using end_time as fallback. "
                "This may indicate an issue with the response format or no tokens were received."
            )
        
        logger.debug(
            f"Generated text: {generated_text} \n"
            f"Time at first token: {time_at_first_token} \n"
            f"Finish reason: {finish_reason}\n"
            f"Completion Tokens: {tokens_received}\n"
            f"Start Time: {start_time}\n"
            f"End Time: {end_time}"
        )
        # Log if token count was not captured accurately
        if not tokens_received:
            tokens_received = len(generated_text.split())

        # Handle None num_prefill_tokens (common for vision requests)
        # Estimate from text prompt using tokenizer if available, otherwise use word count
        if num_prefill_tokens is None:
            # Extract text from messages array (COHEREV2 format) or message field (old format)
            text_prompt = ""
            if request.chat_request:
                if isinstance(request.chat_request, GenericChatRequest):
                    # COHEREV2 format: extract text from messages array
                    if hasattr(request.chat_request, "messages") and request.chat_request.messages:
                        # Get all text content from all messages
                        text_parts = []
                        for msg in request.chat_request.messages:
                            if isinstance(msg.get("content"), list):
                                for content_item in msg["content"]:
                                    if isinstance(content_item, dict) and content_item.get("type") == "TEXT":
                                        text_parts.append(content_item.get("text", ""))
                        text_prompt = " ".join(text_parts)
                elif hasattr(request.chat_request, "message"):
                    # Old CohereChatRequest format
                    text_prompt = request.chat_request.message
            if hasattr(self, "environment") and hasattr(self.environment, "sampler"):
                try:
                    num_prefill_tokens = self.environment.sampler.get_token_length(
                        text_prompt, add_special_tokens=False
                    )
                except Exception:
                    # Fallback to word count estimation if tokenizer fails
                    num_prefill_tokens = len(text_prompt.split())
            else:
                # Fallback to word count estimation
                num_prefill_tokens = len(text_prompt.split())

        return UserChatResponse(
            status_code=200,
            generated_text=generated_text,
            tokens_received=tokens_received,
            time_at_first_token=time_at_first_token,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
        )

    def parse_embedding_response(
        self,
        request: EmbedTextDetails,
        _: EmbedTextResult,
        start_time: float,
        num_prefill_tokens: int,
        end_time: float,
    ) -> UserResponse:
        """
        Parses a non-streaming response.

        Args:
            request (EmbedTextDetails): The request object.
            _ (EmbedTextResult): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): Number of tokens in the prefill/prompt.
            end_time(float): The time when the request was finished.

        Returns:
            UserResponse: A response object with metrics.
        """
        if num_prefill_tokens is None:
            num_prefill_tokens = len(request.inputs)

        return UserResponse(
            status_code=200,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
            time_at_first_token=end_time,
        )

    def parse_rerank_response(
        self,
        request: RerankTextDetails,
        _: RerankTextDetails,
        start_time: float,
        num_prefill_tokens: int,
        end_time: float,
    ) -> UserResponse:
        """
        Parses a non-streaming response.

        Args:
            request (RerankTextDetails): The request object.
            _ (RerankTextDetails): The response object.
            start_time (float): The time when the request was started.
            num_prefill_tokens (int): Number of tokens in the prefill/prompt.
            end_time(float): The time when the request was finished.

        Returns:
            UserResponse: A response object with metrics.
        """

        return UserResponse(
            status_code=200,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
            time_at_first_token=end_time,
        )
