import os
from typing import List, Tuple
from groq import Groq

class LLMClient:
    
    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in your .env file.")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
    
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Tuple[str, str, float]],
        language: str = "English"
    ) -> str:
        
        if not retrieved_docs:
            if language == "Hindi":
                return "माफ़ करें, मुझे इस विषय पर मेरे दस्तावेज़ों में कोई जानकारी नहीं मिली।"
            return "I don't have reliable information on this in my documents. Please check the official government website directly."
        
        context = ""
        for doc_name, text, score in retrieved_docs:
            context += f"\n[Source: {doc_name} | Relevance: {score:.2f}]\n{text}\n"
        
        prompt = f"""You are a helpful assistant for Indian government services.
You have access to official government documents provided below.
Answer the user's question using the information in these documents.
If the documents contain relevant information, use it and cite the source PDF.
If the documents do not contain enough information to fully answer, say:
"My documents have limited information on this. Here is what I found: [share what's relevant]"
Only if the documents contain absolutely nothing relevant, say:
"I don't have this information in my documents. Please visit the official website."
Do NOT make up steps or processes not mentioned in the documents.
Respond in {language}.

Documents:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.1      # change from 0.3 to 0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {e}. Please try again."