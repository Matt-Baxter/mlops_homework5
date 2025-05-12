from fastapi import APIRouter, HTTPException
from src.retriever import retriever
from src.models.query import RAGRequest, RAGResponse, RAGLLMResponse
from src.utils import llm
import os

router = APIRouter()

@router.post("/similar_responses", response_model=RAGResponse)
def get_similar_responses(request: RAGRequest):
    results = retriever.get_similar_responses(request.question)
    return RAGResponse(answers=results)

@router.post("/answer", response_model=RAGLLMResponse)
def get_answer(request: RAGRequest):
    # Get relevant excerpts
    context = retriever.get_similar_responses(request.question)
    
    # Generate an answer using the LLM
    answer = llm.generate_answer(request.question, context)
    
    return RAGLLMResponse(context=context, answer=answer)