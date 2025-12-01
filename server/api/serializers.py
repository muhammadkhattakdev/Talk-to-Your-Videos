from rest_framework import serializers


class ChatRequestSerializer(serializers.Serializer):
    """
    Serializer for incoming chat requests.
    
    What is a Serializer?
    - A serializer converts complex data (like Python objects) into JSON format
    - It also validates incoming data to ensure it meets our requirements
    - Think of it as a "data contract" - it defines what data we expect
    
    Why do we need this?
    - To ensure the frontend sends us the correct data format
    - To validate data before processing (security and reliability)
    - To provide clear error messages if data is invalid
    """
    
    # Define the fields we expect in the request
    
    video_id = serializers.CharField(
        max_length=20,  # YouTube video IDs are 11 characters, we allow some buffer
        required=True,  # This field is mandatory
        help_text="YouTube video ID (e.g., 'dQw4w9WgXcQ')"
    )
    
    question = serializers.CharField(
        max_length=1000,  # Limit question length to prevent abuse
        required=True,
        help_text="User's question about the video",
        allow_blank=False,  # Don't allow empty strings
        trim_whitespace=True  # Remove leading/trailing spaces
    )
    
    def validate_video_id(self, value):
        """
        Custom validation for video_id field.
        
        This method is automatically called by Django REST Framework
        when validating the video_id field.
        
        Why validate?
        - To ensure the video ID has a reasonable format
        - To prevent malicious input
        - To provide helpful error messages
        """
        # Check if video_id is at least 10 characters (YouTube IDs are 11)
        if len(value) < 10:
            raise serializers.ValidationError(
                "Video ID seems too short. Please provide a valid YouTube video ID."
            )
        
        # Return the validated value
        return value
    
    def validate_question(self, value):
        """
        Custom validation for question field.
        """
        # Ensure question is not just whitespace
        if not value.strip():
            raise serializers.ValidationError(
                "Question cannot be empty."
            )
        
        # Ensure question is at least 2 characters
        if len(value.strip()) < 2:
            raise serializers.ValidationError(
                "Question is too short. Please ask a more detailed question."
            )
        
        return value


class ChatResponseSerializer(serializers.Serializer):
    """
    Serializer for outgoing chat responses.
    
    This defines the structure of the data we send back to the frontend.
    It ensures consistency in our API responses.
    """
    
    answer = serializers.CharField(
        help_text="AI-generated answer to the user's question"
    )
    
    video_id = serializers.CharField(
        help_text="The video ID that was queried"
    )
    
    # Optional: Add timestamp to track when response was generated
    # timestamp = serializers.DateTimeField(
    #     default=serializers.CreateOnlyDefault(timezone.now)
    # )


class ErrorResponseSerializer(serializers.Serializer):
    """
    Serializer for error responses.
    
    Provides a consistent error format across the API.
    """
    
    error = serializers.CharField(
        help_text="Brief error message"
    )
    
    detail = serializers.CharField(
        required=False,
        help_text="Detailed error information (for debugging)"
    )