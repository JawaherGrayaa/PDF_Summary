from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pathlib import Path
import google.genai as genai
import pypdf
import io
import os

app = FastAPI()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    # Read the PDF
    contents = await file.read()
    pdf_reader = pypdf.PdfReader(io.BytesIO(contents))

    # Extract text from all pages
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    if not text.strip():
        return {"summary": "Could not extract text from this PDF."}

    # Send to Gemini for summarization
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Please summarize the following document clearly and concisely. Extract the key points and main ideas:\n\n{text[:8000]}"
    )

    return {"summary": response.text}