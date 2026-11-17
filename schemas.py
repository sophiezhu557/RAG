from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

RAGStatus = Literal["Success",
                    "CompanyNotFound",
                    "NoRelevantChunks",
                    "NoAnswerFound"]

class RAGRequest(BaseModel):
    question: str = Field(min_length = 1)

class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    text: str | None
    page: int
    metadata: dict=Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    page: int | None 
    text: str
    id: int
    score: float 

class RAGResponse(BaseModel):
    final_answer: str 
    company_query_list: list[dict]
    selected_chunks: dict[str, list[RetrievedChunk]]

class RAGResponse(BaseModel):
    status: RAGStatus
    message: str | None = None
    final_answer: str | None = None
    company_query_list: list[dict] = Field(default_factory=list)
    selected_chunks: dict[str, list[RetrievedChunk]] = Field(default_factory=dict)
    