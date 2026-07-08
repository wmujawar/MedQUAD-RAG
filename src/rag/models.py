from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    is_relevant: bool = Field(
        description="True if retrieved documents are relevant else False"
    )


class HallucinationCheck(BaseModel):
    is_grounded: bool = Field(description="True if response is grounded else False")
