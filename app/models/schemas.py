from pydantic import BaseModel

class UploadResponse(BaseModel):
    message: str
    filename: str
    file_id: str | None = None

class ErrorResponse(BaseModel):
    error: str