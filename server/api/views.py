from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import ChatRequestSerializer, ChatResponseSerializer
from .services.rag_service import RAGService
import logging

# Set up logging to track what's happening in our application
logger = logging.getLogger(__name__)

# Initialize the RAG service (Retrieval-Augmented Generation)
# This service handles all the AI logic for our chatbot
rag_service = RAGService()


@api_view(['POST'])
def chat_with_video(request):
    """
    API endpoint that handles chat requests about YouTube videos.
    
    This function receives a video ID and a question from the frontend,
    then uses RAG to generate an intelligent response based on the video's transcript.
    
    Flow:
    1. Validate incoming data
    2. Get or process the video transcript
    3. Generate an AI response using RAG
    4. Return the response to the frontend
    """
    
    # Validate the incoming request data using our serializer
    serializer = ChatRequestSerializer(data=request.data)
    
    if not serializer.is_valid():
        # If validation fails, return error messages
        return Response(
            serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Extract validated data
    video_id = serializer.validated_data['video_id']
    question = serializer.validated_data['question']
    
    try:
        # Call the RAG service to get an answer
        # This is where the magic happens - RAG retrieves relevant parts
        # of the transcript and generates a contextual answer
        answer = rag_service.get_answer(video_id, question)
        
        # Prepare the response data
        response_data = {
            'answer': answer,
            'video_id': video_id
        }
        
        # Validate the response data
        response_serializer = ChatResponseSerializer(data=response_data)
        
        if response_serializer.is_valid():
            return Response(
                response_serializer.validated_data,
                status=status.HTTP_200_OK
            )
        
        # If response validation fails (shouldn't happen normally)
        return Response(
            response_serializer.errors,
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Error processing chat request: {str(e)}")
        
        # Return a user-friendly error message
        return Response(
            {
                'error': 'Failed to process your question. Please ensure the video has captions available.',
                'detail': str(e)
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def health_check(request):
    """
    Simple endpoint to check if the API is running.
    Useful for monitoring and debugging.
    """
    return Response(
        {'status': 'ok', 'message': 'API is running'},
        status=status.HTTP_200_OK
    )