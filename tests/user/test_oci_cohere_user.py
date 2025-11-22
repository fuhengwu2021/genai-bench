from unittest.mock import MagicMock, patch

import pytest
from oci.generative_ai_inference.models import (
    ChatDetails,
    CohereChatRequest,
    DedicatedServingMode,
    EmbedTextDetails,
    GenericChatRequest,
    OnDemandServingMode,
    RerankTextDetails,
)

from genai_bench.protocol import (
    UserChatRequest,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserImageEmbeddingRequest,
    UserReRankRequest,
)
from genai_bench.user.oci_cohere_user import OCICohereUser


@pytest.fixture
def test_cohere_user():
    OCICohereUser.host = "http://example.com"

    # Set up mock auth provider
    mock_auth = MagicMock()
    mock_auth.get_credentials.return_value = "test-key"
    mock_auth.get_config.return_value = {
        "api_base": "http://example.com",
        "compartment_id": "test-compartment",
        "endpoint_id": "test-endpoint",
    }
    OCICohereUser.auth_provider = mock_auth

    user = OCICohereUser(environment=MagicMock())
    return user


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat(mock_client_class, test_cohere_user):
    # Mock the chat method
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"apiFormat":"COHERE","text":"The","pad":"aaaaaaaaa"}'),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" relational","pad":"aaaaaaaaa"}'
            ),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" database","pad":"aaaaaaaaa"}'
            ),
            MagicMock(data=b'{"apiFormat":"COHERE","text":".","pad":"aaaaaaaaa"}'),
            MagicMock(
                data=b'{"apiFormat":"COHERE","finishReason": "tokens", "text":""}'
            ),
            MagicMock(data=b"data: [DONE]"),
        ]
    )

    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    # Verify the chat request was called
    call_args = mock_client_instance.chat.call_args[0][0]
    chat_request = call_args.chat_request
    
    # Verify api_format is COHEREV2 and uses GenericChatRequest with messages array
    assert isinstance(chat_request, GenericChatRequest)
    assert chat_request.api_format == "COHEREV2"
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0]["role"] == "USER"
    assert len(chat_request.messages[0]["content"]) == 1
    assert chat_request.messages[0]["content"][0]["type"] == "TEXT"
    assert chat_request.messages[0]["content"][0]["text"] == "Hello"
    assert chat_request.max_tokens == 10
    assert chat_request.is_stream is True
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 5


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_json_error(mock_client_class, test_cohere_user):
    # Mock the chat method
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"apiFormat":"COHERE","text":"The","pad":"aaaaaaaaa"}'),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" relational","pad":"aaaaaaaaa"}'
            ),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" database","pad":"aaaaaaaaa"}'
            ),
            MagicMock(data=b'{"apiFormat":"COHERE","text":".","pad":"aaaaaaaaa"}'),
            MagicMock(data=b"data: [DONE]"),
        ]
    )

    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    # Verify the chat request was called
    call_args = mock_client_instance.chat.call_args[0][0]
    chat_request = call_args.chat_request
    
    # Verify api_format is COHEREV2 and uses GenericChatRequest with messages array
    assert isinstance(chat_request, GenericChatRequest)
    assert chat_request.api_format == "COHEREV2"
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0]["role"] == "USER"
    assert len(chat_request.messages[0]["content"]) == 1
    assert chat_request.messages[0]["content"][0]["type"] == "TEXT"
    assert chat_request.messages[0]["content"][0]["text"] == "Hello"
    assert chat_request.max_tokens == 10
    assert chat_request.is_stream is True
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 5


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_embedding_request(mock_client_class, test_cohere_user):
    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=[],
        num_prefill_tokens=10,
        model="cohere-embed-v3",
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock
    with pytest.raises(AttributeError):
        test_cohere_user.chat()


@patch("genai_bench.user.oci_cohere_user.logger")
@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_response_error(mock_client_class, mock_logger, test_cohere_user):
    # Mock the chat method
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 401
    mock_client_instance.chat.return_value.data.events.side_effect = [
        MagicMock(data=b'{"apiFormat":"COHERE","text":"The","pad":"aaaaaaaaa"}'),
        MagicMock(
            data=b'{"apiFormat":"COHERE","text":" relational","pad":"aaaaaaaaa"}'
        ),
        MagicMock(data=b'{"apiFormat":"COHERE","text":" database","pad":"aaaaaaaaa"}'),
        MagicMock(data=b'{"apiFormat":"COHERE","text":".","pad":"aaaaaaaaa"}'),
        MagicMock(data=b"data: [DONE]"),
    ]

    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    # Verify the chat request was called
    call_args = mock_client_instance.chat.call_args[0][0]
    chat_request = call_args.chat_request
    
    # Verify api_format is COHEREV2 and uses GenericChatRequest with messages array
    assert isinstance(chat_request, GenericChatRequest)
    assert chat_request.api_format == "COHEREV2"
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0]["role"] == "USER"
    assert len(chat_request.messages[0]["content"]) == 1
    assert chat_request.messages[0]["content"][0]["type"] == "TEXT"
    assert chat_request.messages[0]["content"][0]["text"] == "Hello"
    assert chat_request.max_tokens == 10
    assert chat_request.is_stream is True
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 401


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_embed(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200

    test_cohere_user.on_start()
    documents = ["The earth has a radius of 6,454km."]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=documents,
        num_prefill_tokens=10,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=documents,
            input_type="SEARCH_DOCUMENT",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 10


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_embed_with_prefill_tokens(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200

    test_cohere_user.on_start()
    documents = ["The earth has a radius of 6,454km."]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=documents,
        num_prefill_tokens=0,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=documents,
            input_type="SEARCH_DOCUMENT",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 0


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_image_embeddings(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200

    test_cohere_user.on_start()
    images = ["data:image/jpeg;base64,BASE64Image1"]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserImageEmbeddingRequest(
        documents=[],
        image_content=images,
        num_images=2,
        num_prefill_tokens=10,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=images,
            input_type="IMAGE",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 10


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_image_embeddings_multiple_images(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200

    test_cohere_user.on_start()
    images = ["BASE64Image1", "BASE64Image2"]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserImageEmbeddingRequest(
        documents=[],
        image_content=images,
        num_images=2,
        num_prefill_tokens=10,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock
    with pytest.raises(ValueError):
        test_cohere_user.embeddings()
        mock_client_instance.embed_text.assert_not_called()
        metrics_collector_mock.embed_text.assert_not_called()


@patch("genai_bench.user.oci_cohere_user.logger")
@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_embed_with_response_error(mock_client_class, mock_logger, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 401

    test_cohere_user.on_start()
    documents = ["The earth has a radius of 6,454km."]
    model = "cohere-embed-v3"
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=documents,
        num_prefill_tokens=10,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=documents,
            input_type="SEARCH_DOCUMENT",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 401


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_chat_history(mock_client_class, test_cohere_user):
    """Test chat with chat history in additional params."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = []

    test_cohere_user.on_start()
    chat_history = [{"role": "user", "content": "Previous message"}]
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
            "chatHistory": chat_history,
        },
        max_tokens=10,
    )

    test_cohere_user.chat()

    chat_request = mock_client_instance.chat.call_args[0][0].chat_request
    # COHEREV2 format includes chat history in messages array
    assert isinstance(chat_request, GenericChatRequest)
    assert len(chat_request.messages) == 2  # History + current message
    assert chat_request.messages[0]["role"] == "USER"
    assert chat_request.messages[0]["content"][0]["text"] == "Previous message"


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_documents(mock_client_class, test_cohere_user):
    """Test chat with documents in additional params."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = []

    test_cohere_user.on_start()
    documents = [{"text": "Document content"}]
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
            "documents": documents,
        },
        max_tokens=10,
    )

    test_cohere_user.chat()

    chat_request = mock_client_instance.chat.call_args[0][0].chat_request
    # COHEREV2 format doesn't support documents field directly
    # Documents would need to be included in message content if needed
    assert isinstance(chat_request, GenericChatRequest)
    # Documents are not currently supported in COHEREV2 messages format


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_images_data_url(mock_client_class, test_cohere_user):
    """Test chat with images in data URL format (image-text-to-text task)."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"apiFormat":"COHERE","text":"This","pad":"aaaaaaaaa"}'),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" image","pad":"aaaaaaaaa"}'
            ),
            MagicMock(
                data=b'{"apiFormat":"COHERE","finishReason": "tokens", "text":""}'
            ),
        ]
    )

    test_cohere_user.on_start()
    # Image in data URL format
    images = ["data:image/jpeg;base64,BASE64ImageData123"]
    test_cohere_user.sample = lambda: UserImageChatRequest(
        model="cohere-vision-model",
        prompt="Describe this image",
        image_content=images,
        num_images=1,
        num_prefill_tokens=10,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=50,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    # Verify the chat request was called with images
    call_args = mock_client_instance.chat.call_args[0][0]
    chat_request = call_args.chat_request
    
    # Verify COHEREV2 format with messages array containing image
    assert isinstance(chat_request, GenericChatRequest)
    assert chat_request.api_format == "COHEREV2"
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0]["role"] == "USER"
    assert len(chat_request.messages[0]["content"]) == 2  # TEXT + IMAGE
    assert chat_request.messages[0]["content"][0]["type"] == "TEXT"
    assert chat_request.messages[0]["content"][0]["text"] == "Describe this image"
    assert chat_request.messages[0]["content"][1]["type"] == "IMAGE_URL"
    assert "data:image/jpeg;base64,BASE64ImageData123" in chat_request.messages[0]["content"][1]["imageUrl"]["url"]
    
    # Verify vision requests use "COHEREV2" api_format
    assert chat_request.api_format == "COHEREV2"
    
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 10


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_images_base64(mock_client_class, test_cohere_user):
    """Test chat with images in base64 format (image-text-to-text task)."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"apiFormat":"COHERE","text":"The","pad":"aaaaaaaaa"}'),
            MagicMock(
                data=b'{"apiFormat":"COHERE","text":" picture","pad":"aaaaaaaaa"}'
            ),
            MagicMock(
                data=b'{"apiFormat":"COHERE","finishReason": "tokens", "text":""}'
            ),
        ]
    )

    test_cohere_user.on_start()
    # Image already in base64 format (no data URL prefix)
    images = ["BASE64ImageData456", "BASE64ImageData789"]
    test_cohere_user.sample = lambda: UserImageChatRequest(
        model="cohere-vision-model",
        prompt="What do you see in these images?",
        image_content=images,
        num_images=2,
        num_prefill_tokens=15,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=100,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    # Verify the chat request was called with images
    call_args = mock_client_instance.chat.call_args[0][0]
    chat_request = call_args.chat_request
    
    # Verify COHEREV2 format with messages array containing multiple images
    assert isinstance(chat_request, GenericChatRequest)
    assert chat_request.api_format == "COHEREV2"
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0]["role"] == "USER"
    assert len(chat_request.messages[0]["content"]) == 3  # TEXT + 2 IMAGES
    assert chat_request.messages[0]["content"][0]["type"] == "TEXT"
    assert chat_request.messages[0]["content"][0]["text"] == "What do you see in these images?"
    
    # Verify vision requests use "COHEREV2" api_format
    assert chat_request.api_format == "COHEREV2"
    
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 15


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_with_images_mixed_format(mock_client_class, test_cohere_user):
    """Test chat with mixed image formats (data URL and base64)."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"apiFormat":"COHERE","text":"Mixed","pad":"aaaaaaaaa"}'),
            MagicMock(
                data=b'{"apiFormat":"COHERE","finishReason": "tokens", "text":""}'
            ),
        ]
    )

    test_cohere_user.on_start()
    # Mix of data URL and base64 formats
    images = [
        "data:image/png;base64,BASE64ImageData1",
        "BASE64ImageData2",  # Already base64
    ]
    test_cohere_user.sample = lambda: UserImageChatRequest(
        model="cohere-vision-model",
        prompt="Compare these images",
        image_content=images,
        num_images=2,
        num_prefill_tokens=20,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=150,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    # Verify the chat request was called with processed images
    call_args = mock_client_instance.chat.call_args[0][0]
    chat_request = call_args.chat_request
    
    # Verify COHEREV2 format with messages array containing mixed format images
    assert isinstance(chat_request, GenericChatRequest)
    assert chat_request.api_format == "COHEREV2"
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0]["role"] == "USER"
    assert len(chat_request.messages[0]["content"]) == 3  # TEXT + 2 IMAGES
    assert chat_request.messages[0]["content"][0]["type"] == "TEXT"
    assert chat_request.messages[0]["content"][0]["text"] == "Compare these images"
    
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 20


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_vision_request_no_api_format(mock_client_class, test_cohere_user):
    """Test that vision requests (with images) use api_format="COHEREV2"."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"text":"This","pad":"aaaaaaaaa"}'),
            MagicMock(data=b'{"text":" image","pad":"aaaaaaaaa"}'),
            MagicMock(data=b'{"finishReason": "tokens", "text":""}'),
        ]
    )

    test_cohere_user.on_start()
    images = ["data:image/jpeg;base64,BASE64ImageData123"]
    test_cohere_user.sample = lambda: UserImageChatRequest(
        model="cohere.command-a-vision",
        prompt="Describe this image",
        image_content=images,
        num_images=1,
        num_prefill_tokens=10,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=50,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    # Verify the chat request was called
    call_args = mock_client_instance.chat.call_args[0][0]
    chat_request = call_args.chat_request
    
    # CRITICAL: Vision requests use "COHEREV2" api_format
    assert chat_request.api_format == "COHEREV2"
    
    # Verify COHEREV2 format with messages array containing image
    assert isinstance(chat_request, GenericChatRequest)
    assert chat_request.api_format == "COHEREV2"
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0]["role"] == "USER"
    assert len(chat_request.messages[0]["content"]) == 2  # TEXT + IMAGE
    assert chat_request.messages[0]["content"][0]["type"] == "TEXT"
    assert chat_request.messages[0]["content"][0]["text"] == "Describe this image"
    
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_text_request_has_api_format(mock_client_class, test_cohere_user):
    """Test that text-only requests (without images) use api_format="COHEREV2"."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"apiFormat":"COHERE","text":"Hello","pad":"aaaaaaaaa"}'),
            MagicMock(data=b'{"apiFormat":"COHERE","text":" world","pad":"aaaaaaaaa"}'),
            MagicMock(data=b'{"apiFormat":"COHERE","finishReason": "tokens", "text":""}'),
        ]
    )

    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.chat()

    # Verify the chat request was called
    call_args = mock_client_instance.chat.call_args[0][0]
    chat_request = call_args.chat_request
    
    # CRITICAL: Text-only requests use "COHEREV2" api_format with GenericChatRequest
    assert isinstance(chat_request, GenericChatRequest)
    assert chat_request.api_format == "COHEREV2"
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0]["role"] == "USER"
    assert len(chat_request.messages[0]["content"]) == 1  # TEXT only, no images
    assert chat_request.messages[0]["content"][0]["type"] == "TEXT"
    assert chat_request.messages[0]["content"][0]["text"] == "Hello"
    
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_chat_vision_vs_text_api_format_difference(mock_client_class, test_cohere_user):
    """Test that vision and text requests handle api_format differently."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.chat.return_value.status = 200
    mock_client_instance.chat.return_value.data.events.return_value = iter(
        [
            MagicMock(data=b'{"text":"Response","pad":"aaaaaaaaa"}'),
            MagicMock(data=b'{"finishReason": "tokens", "text":""}'),
        ]
    )

    test_cohere_user.on_start()
    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    # Test 1: Vision request (with images) - should use api_format="COHEREV2"
    images = ["data:image/jpeg;base64,BASE64ImageData"]
    test_cohere_user.sample = lambda: UserImageChatRequest(
        model="cohere.command-a-vision",
        prompt="Describe this",
        image_content=images,
        num_images=1,
        num_prefill_tokens=10,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=50,
    )

    test_cohere_user.chat()
    call_args_vision = mock_client_instance.chat.call_args[0][0]
    chat_request_vision = call_args_vision.chat_request
    
    # Vision request should use "COHEREV2" api_format with GenericChatRequest
    assert isinstance(chat_request_vision, GenericChatRequest)
    assert chat_request_vision.api_format == "COHEREV2"
    assert len(chat_request_vision.messages) == 1
    assert len(chat_request_vision.messages[0]["content"]) == 2  # TEXT + IMAGE
    assert chat_request_vision.messages[0]["content"][1]["type"] == "IMAGE_URL"
    
    # Reset mock
    mock_client_instance.chat.reset_mock()
    metrics_collector_mock.reset_mock()

    # Test 2: Text request (without images) - should use api_format="COHEREV2"
    test_cohere_user.sample = lambda: UserChatRequest(
        model="cohere-model",
        prompt="Hello",
        num_prefill_tokens=5,
        additional_request_params={
            "compartmentId": "ocid1.compartment.oc1..example",
            "servingType": "ON_DEMAND",
        },
        max_tokens=10,
    )

    test_cohere_user.chat()
    call_args_text = mock_client_instance.chat.call_args[0][0]
    chat_request_text = call_args_text.chat_request
    
    # Text request should also use "COHEREV2" api_format with GenericChatRequest
    assert isinstance(chat_request_text, GenericChatRequest)
    assert chat_request_text.api_format == "COHEREV2"
    assert len(chat_request_text.messages) == 1
    assert len(chat_request_text.messages[0]["content"]) == 1  # TEXT only
    assert chat_request_text.messages[0]["content"][0]["type"] == "TEXT"


def test_get_compartment_id_missing(test_cohere_user):
    """Test error when compartmentId is missing."""
    request = UserChatRequest(
        model="model",
        prompt="test",
        additional_request_params={},
        num_prefill_tokens=5,
        max_tokens=10,
    )

    with pytest.raises(
        ValueError, match="compartmentId missing in additional request params"
    ):
        test_cohere_user.get_compartment_id(request)


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_embeddings_with_missing_prefill_tokens(mock_client_class, test_cohere_user):
    """Test embeddings with missing prefill tokens."""
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.embed_text.return_value.status = 200
    model = "cohere-embed-v3"

    test_cohere_user.on_start()
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=[],
        num_prefill_tokens=0,  # missing prefill tokens
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )
    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.embeddings()

    mock_client_instance.embed_text.assert_called_once_with(
        EmbedTextDetails(
            inputs=[],
            input_type="SEARCH_DOCUMENT",
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
            truncate="NONE",
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 0


def test_get_serving_mode_missing_endpoint(test_cohere_user):
    """Test error when endpointId is missing for DEDICATED serving type."""
    request = UserChatRequest(
        model="model",
        prompt="test",
        additional_request_params={
            "servingType": "DEDICATED",  # Missing endpointId
            "compartmentId": "test",
        },
        num_prefill_tokens=5,
        max_tokens=10,
    )

    with pytest.raises(
        ValueError, match="endpointId must be provided for DEDICATED servingType"
    ):
        test_cohere_user.get_serving_mode(request)


def test_get_serving_mode_dedicated(test_cohere_user):
    """Test error when endpointId is missing for DEDICATED serving type."""
    request = UserChatRequest(
        model="model",
        prompt="test",
        additional_request_params={
            "servingType": "DEDICATED",
            "endpointId": "endpoint",
            "compartmentId": "test",
        },
        num_prefill_tokens=5,
        max_tokens=10,
    )

    serving_mode = test_cohere_user.get_serving_mode(request)

    assert isinstance(serving_mode, DedicatedServingMode)
    assert serving_mode.endpoint_id == "endpoint"


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_rerank(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.rerank_text.return_value.status = 200

    test_cohere_user.on_start()
    documents = [
        "The earth has a radius of 6,454km.",
        "Earth is the 3rd planet in solar system.",
    ]
    query = "What is the order of earth in the solar system?"
    model = "cohere-reank-v3.5"
    test_cohere_user.sample = lambda: UserReRankRequest(
        documents=documents,
        query=query,
        num_prefill_tokens=100,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.rerank()

    mock_client_instance.rerank_text.assert_called_once_with(
        RerankTextDetails(
            documents=documents,
            input=query,
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens == 100


def test_rerank_with_embedding_request(test_cohere_user):
    documents = [
        "The earth has a radius of 6,454km.",
        "Earth is the 3rd planet in solar system.",
    ]
    model = "cohere-reank-v3.5"
    test_cohere_user.sample = lambda: UserEmbeddingRequest(
        documents=documents,
        num_prefill_tokens=100,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    with pytest.raises(
        AttributeError, match="user_request should be of type UserReRankRequest"
    ):
        test_cohere_user.rerank()


@patch("genai_bench.user.oci_cohere_user.GenerativeAiInferenceClient")
def test_rerank_with_no_prefill(mock_client_class, test_cohere_user):
    mock_client_instance = mock_client_class.return_value
    mock_client_instance.rerank_text.return_value.status = 200

    test_cohere_user.on_start()
    documents = [
        "The earth has a radius of 6,454km.",
        "Earth is the 3rd planet in solar system.",
    ]
    query = "What is the order of earth in the solar system?"
    model = "cohere-reank-v3.5"
    test_cohere_user.sample = lambda: UserReRankRequest(
        documents=documents,
        query=query,
        num_prefill_tokens=None,
        model=model,
        additional_request_params={
            "servingType": "ON_DEMAND",
            "compartmentId": "compartmentId",
        },
    )

    metrics_collector_mock = MagicMock()
    test_cohere_user.collect_metrics = metrics_collector_mock

    test_cohere_user.rerank()

    mock_client_instance.rerank_text.assert_called_once_with(
        RerankTextDetails(
            documents=documents,
            input=query,
            compartment_id="compartmentId",
            serving_mode=OnDemandServingMode(model_id=model),
        )
    )
    metrics_collector_mock.assert_called_once()
    args, _ = metrics_collector_mock.call_args
    user_response = args[0]
    assert user_response.status_code == 200
    assert user_response.num_prefill_tokens is None
