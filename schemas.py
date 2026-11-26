from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class TokenUsage(BaseModel):
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    total_tokens: int | None = 0

    @classmethod
    def from_api(cls, usage):
        if usage is None:
            return cls(input_tokens=None, output_tokens=None, total_tokens=None)

        return cls(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )



RAGStatus = Literal["Success",
                    "NoRelevantChunks",
                    "CompanyNotFound",
                    "NoAnswerFound",
                    "MissingIndex"]

class RAGRequest(BaseModel):
    question: str = Field(min_length = 1)

class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    text: str = Field(min_length=1)
    page: int
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    sha1: str
    page: int | None
    text: str
    id: int
    score: float

class RAGResponse(BaseModel):
    status: RAGStatus
    rewritten_query: str | None
    message: str | None = None
    final_answer: str | None = None
    company_query_list: list[dict] = Field(default_factory=list)
    selected_chunks: dict[str, list[RetrievedChunk]] = Field(default_factory=dict)
    rewrite_usage: TokenUsage = Field(default_factory=TokenUsage)
    generation_usage: TokenUsage = Field(default_factory=TokenUsage)
    