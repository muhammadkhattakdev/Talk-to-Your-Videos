import os
from typing import List, Dict, Optional
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from django.core.cache import cache
import numpy as np

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


class RAGService:

    def __init__(self):

        self.llm_model = genai.GenerativeModel('gemini-pro')

        self.embedding_model_name = 'models/embedding-001'

        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.8,        # Nucleus sampling parameter
            'top_k': 40,         # Consider top K tokens
            'max_output_tokens': 1024,
        }

        self.chunk_size = 1000

        self.chunk_overlap = 200

        self.top_k_chunks = 3

    def get_answer(self, video_id: str, question: str) -> str:

        transcript = self._get_transcript(video_id)

        chunks = self._create_chunks(transcript)

        chunk_embeddings = self._create_embeddings(chunks)
        question_embedding = self._create_embedding(question)

        relevant_chunks = self._find_relevant_chunks(
            chunks,
            chunk_embeddings,
            question_embedding
        )

        answer = self._generate_answer(question, relevant_chunks)

        return answer

    def _get_transcript(self, video_id: str) -> str:

        cache_key = f'transcript_{video_id}'

        cached_transcript = cache.get(cache_key)
        if cached_transcript:
            return cached_transcript

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            full_transcript = ' '.join([entry['text'] for entry in transcript_list])

            cache.set(cache_key, full_transcript, timeout=3600)

            return full_transcript

        except Exception as e:
            raise Exception(f"Failed to fetch transcript: {str(e)}")

    def _create_chunks(self, text: str) -> List[str]:

        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            chunk = text[start:end]

            if chunk.strip():  # .strip() removes whitespace
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:


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
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"  # Optimized for document retrieval
            )
            
            return result['embedding']
            
        except Exception as e:
            raise Exception(f"Failed to create embedding: {str(e)}")
    
    def _find_relevant_chunks(
        self,
        chunks: List[str],
        chunk_embeddings: List[List[float]],
        question_embedding: List[float]
    ) -> List[str]:

        
        similarities = []
        
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = self._cosine_similarity(
                question_embedding,
                chunk_embedding
            )
            similarities.append((i, similarity))  # Store index and similarity
        

        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K chunks
        top_indices = [idx for idx, _ in similarities[:self.top_k_chunks]]
        relevant_chunks = [chunks[i] for i in top_indices]
        
        return relevant_chunks
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return float(similarity)  # Convert numpy type to Python float
    
    def _generate_answer(self, question: str, relevant_chunks: List[str]) -> str:

        context = '\n\n'.join(relevant_chunks)
        
        prompt = f"""You are a helpful assistant that answers questions about YouTube videos.

Context from the video transcript:
{context}

User's question: {question}

Please provide a clear, accurate answer based on the context above. If the context doesn't contain relevant information to answer the question, say so politely.

Answer:"""
        
        try:
            response = self.llm_model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Failed to generate answer: {str(e)}")


