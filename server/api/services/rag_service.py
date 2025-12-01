"""
RAG Service - The Brain of Our ChatBot

This file contains the RAGService class which implements Retrieval-Augmented Generation.

What is RAG?
RAG = Retrieval-Augmented Generation
It's a technique that combines:
1. RETRIEVAL: Finding relevant information from a knowledge base
2. GENERATION: Using AI to generate answers based on that information

Why RAG?
- LLMs (like Gemini) are powerful but don't know about specific videos
- RAG allows us to give the LLM context about the video
- It's more efficient than sending the entire transcript every time
"""

import os
from typing import List, Dict, Optional
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from django.core.cache import cache
import numpy as np

# Configure Gemini API
# Make sure to set this in your environment variables or settings.py
genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'your-api-key-here'))


class RAGService:
    """
    RAG Service for YouTube Video Question Answering
    
    This class handles:
    1. Fetching video transcripts
    2. Breaking transcripts into chunks
    3. Creating embeddings (numerical representations) of chunks
    4. Finding relevant chunks for a question
    5. Generating answers using Gemini LLM
    """
    
    def __init__(self):
        """
        Initialize the RAG service with models.
        
        What happens here?
        - We set up the Gemini models we'll use
        - One for generating text (gemini-pro)
        - One for creating embeddings (embedding-001)
        """
        
        # Model for generating conversational responses
        self.llm_model = genai.GenerativeModel('gemini-pro')
        
        # Model name for creating embeddings
        self.embedding_model_name = 'models/embedding-001'
        
        # Configuration for text generation
        self.generation_config = {
            'temperature': 0.7,  # Controls randomness (0=focused, 1=creative)
            'top_p': 0.8,        # Nucleus sampling parameter
            'top_k': 40,         # Consider top K tokens
            'max_output_tokens': 1024,  # Maximum length of response
        }
        
        # Chunk size for splitting transcripts
        # Why 1000? It's a balance between context and precision
        self.chunk_size = 1000
        
        # Overlap between chunks to maintain context
        self.chunk_overlap = 200
        
        # Number of relevant chunks to retrieve
        self.top_k_chunks = 3
    
    def get_answer(self, video_id: str, question: str) -> str:
        """
        Main method to get an answer for a question about a video.
        
        Args:
            video_id (str): YouTube video ID
            question (str): User's question
            
        Returns:
            str: AI-generated answer
            
        What is 'str'?
        - 'str' is Python's type hint for string
        - Type hints help developers understand what types are expected
        - They also help IDEs provide better autocomplete
        """
        
        # Step 1: Get the transcript
        transcript = self._get_transcript(video_id)
        
        # Step 2: Break transcript into chunks
        chunks = self._create_chunks(transcript)
        
        # Step 3: Create embeddings for chunks and question
        chunk_embeddings = self._create_embeddings(chunks)
        question_embedding = self._create_embedding(question)
        
        # Step 4: Find most relevant chunks
        relevant_chunks = self._find_relevant_chunks(
            chunks,
            chunk_embeddings,
            question_embedding
        )
        
        # Step 5: Generate answer using LLM
        answer = self._generate_answer(question, relevant_chunks)
        
        return answer
    
    def _get_transcript(self, video_id: str) -> str:
        """
        Fetch the transcript for a YouTube video.
        
        Why use cache?
        - Transcripts don't change, so we can store them
        - This makes subsequent requests much faster
        - Reduces API calls to YouTube
        
        What is cache?
        - A temporary storage for frequently accessed data
        - Like a notebook where you write down things you look up often
        """
        
        # Create a unique cache key for this video
        cache_key = f'transcript_{video_id}'
        
        # Try to get from cache first
        cached_transcript = cache.get(cache_key)
        if cached_transcript:
            return cached_transcript
        
        try:
            # Fetch transcript from YouTube
            # This returns a list of dictionaries with 'text', 'start', 'duration'
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine all text segments into one string
            # Why join? It's efficient and creates a single string
            full_transcript = ' '.join([entry['text'] for entry in transcript_list])
            
            # Cache for 1 hour (3600 seconds)
            cache.set(cache_key, full_transcript, timeout=3600)
            
            return full_transcript
            
        except Exception as e:
            raise Exception(f"Failed to fetch transcript: {str(e)}")
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Why chunk?
        - Large texts are hard to process at once
        - Embeddings work better on focused content
        - We can retrieve only relevant parts
        
        What is List[str]?
        - List[str] means "a list containing strings"
        - It's a type hint telling us this returns a list of strings
        - Example: ["chunk 1", "chunk 2", "chunk 3"]
        """
        
        chunks = []
        
        # Calculate how many chunks we'll create
        # Why these calculations? To ensure we cover all text with overlap
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Define end of current chunk
            end = start + self.chunk_size
            
            # Extract chunk (substring from start to end)
            chunk = text[start:end]
            
            # Add to list if chunk has content
            if chunk.strip():  # .strip() removes whitespace
                chunks.append(chunk)
            
            # Move start position for next chunk
            # We subtract overlap to maintain context between chunks
            start = end - self.chunk_overlap
        
        return chunks
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        What is an embedding?
        - A numerical representation of text
        - Converts words into a list of numbers (vector)
        - Similar meanings = similar numbers
        - Example: "cat" and "kitten" will have similar embeddings
        
        What is List[List[float]]?
        - A list of lists containing floating-point numbers
        - Each inner list is one embedding (vector)
        - Example: [[0.1, 0.5, 0.3], [0.2, 0.4, 0.6]]
        
        What is float?
        - A number with decimal points (like 3.14, 0.5, 2.0)
        """
        
        embeddings = []
        
        for text in texts:
            embedding = self._create_embedding(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.
        
        This uses Google's embedding model to convert text to numbers.
        """
        
        try:
            # Call Gemini's embedding API
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"  # Optimized for document retrieval
            )
            
            # Return the embedding vector
            return result['embedding']
            
        except Exception as e:
            raise Exception(f"Failed to create embedding: {str(e)}")
    
    def _find_relevant_chunks(
        self,
        chunks: List[str],
        chunk_embeddings: List[List[float]],
        question_embedding: List[float]
    ) -> List[str]:
        """
        Find the most relevant chunks for a question.
        
        How?
        - Calculate similarity between question and each chunk
        - Use cosine similarity (measures angle between vectors)
        - Return top K most similar chunks
        
        Why cosine similarity?
        - It measures how "aligned" two vectors are
        - Perfect alignment = 1.0 (very similar)
        - Opposite = -1.0 (very different)
        - Perpendicular = 0.0 (unrelated)
        """
        
        similarities = []
        
        # Calculate similarity for each chunk
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = self._cosine_similarity(
                question_embedding,
                chunk_embedding
            )
            similarities.append((i, similarity))  # Store index and similarity
        
        # Sort by similarity (highest first)
        # Why sorted? To find the most relevant chunks
        # key=lambda x: x[1] means "sort by the second element (similarity)"
        # reverse=True means "highest first"
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K chunks
        top_indices = [idx for idx, _ in similarities[:self.top_k_chunks]]
        relevant_chunks = [chunks[i] for i in top_indices]
        
        return relevant_chunks
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Mathematical formula:
        similarity = (A · B) / (||A|| * ||B||)
        
        Where:
        - A · B is the dot product (multiply corresponding elements and sum)
        - ||A|| is the magnitude (length) of vector A
        - ||B|| is the magnitude of vector B
        
        Why numpy?
        - NumPy is a library for numerical computing
        - It's much faster than pure Python for math operations
        - Perfect for working with vectors and matrices
        """
        
        # Convert to numpy arrays for efficient computation
        # np.array converts Python list to NumPy array
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate dot product (sum of element-wise multiplication)
        dot_product = np.dot(vec1, vec2)
        
        # Calculate magnitudes (length of vectors)
        # np.linalg.norm calculates the Euclidean norm
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Calculate and return cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return float(similarity)  # Convert numpy type to Python float
    
    def _generate_answer(self, question: str, relevant_chunks: List[str]) -> str:
        """
        Generate an answer using the LLM and relevant context.
        
        This is where we use Gemini to create a natural language response.
        """
        
        # Combine relevant chunks into context
        # '\n\n'.join() puts each chunk on a new line with spacing
        context = '\n\n'.join(relevant_chunks)
        
        # Create a prompt for the LLM
        # A prompt is instructions + context for the AI
        prompt = f"""You are a helpful assistant that answers questions about YouTube videos.

Context from the video transcript:
{context}

User's question: {question}

Please provide a clear, accurate answer based on the context above. If the context doesn't contain relevant information to answer the question, say so politely.

Answer:"""
        
        try:
            # Generate response using Gemini
            response = self.llm_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Extract and return the text
            return response.text
            
        except Exception as e:
            raise Exception(f"Failed to generate answer: {str(e)}")


"""
SUMMARY - How RAG Works in This Code:

1. USER ASKS QUESTION about a video
   ↓
2. GET TRANSCRIPT from YouTube
   ↓
3. SPLIT into chunks (smaller pieces)
   ↓
4. CREATE EMBEDDINGS (convert text to numbers) for all chunks
   ↓
5. CREATE EMBEDDING for the question
   ↓
6. FIND SIMILAR chunks (using cosine similarity)
   ↓
7. SEND question + relevant chunks to Gemini
   ↓
8. GENERATE ANSWER using AI
   ↓
9. RETURN answer to user

Why is this better than just asking Gemini?
- More accurate (uses actual video content)
- More efficient (doesn't send entire transcript)
- More relevant (only uses related parts)
- Better context management (chunk size optimization)
"""