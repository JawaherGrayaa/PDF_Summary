from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pathlib import Path
from groq import Groq
import pypdf
import io
import os

app = FastAPI()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    contents = await file.read()
    pdf_reader = pypdf.PdfReader(io.BytesIO(contents))

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"

    if not text.strip():
        return {"summary": "Could not extract text from this PDF."}

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"Please summarize the following document clearly and concisely. Extract the key points and main ideas:\n\n{text[:8000]}"
            }
        ]
    )

    return {"summary": response.choices[0].message.content}