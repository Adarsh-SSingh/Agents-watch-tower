from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from agentops.agents.document.agent import run_document_agent
from agentops.agents.document.chunker import chunk_documents
from agentops.agents.document.loader import load_uploaded_document
from agentops.agents.document.vector_store import store_documents

router = APIRouter()

# DOCUMENT UPLOAD

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".pptx"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        extension = Path(file.filename or "").suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format."
            )

        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file.")

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds 10 MB."
            )

        documents = load_uploaded_document(
            file_bytes=file_bytes,
            filename=file.filename
        )

        chunks = chunk_documents(documents)
        document_id = store_documents(chunks)

        return {
            "status": "success",
            "document_id": document_id,
            "filename": file.filename,
            "chunks_stored": len(chunks)
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


# DOCUMENT QUESTION

class DocumentQuestion(BaseModel):
    question: str
    document_id: str


@router.post("/documents/ask")
def ask_document(request: DocumentQuestion):
    try:
        result = run_document_agent(
            user_input=request.question,
            document_id=request.document_id,
            top_k=4
        )

        return result

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))