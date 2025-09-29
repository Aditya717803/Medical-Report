from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import easyocr
import logging
from google import genai
import base64
import os
from typing import Optional

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Report Summarizer")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Gemini API client
api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBWyEkl7gxRZnqSKQ2jKZmbHEs9f6XliIE")
client = genai.Client(api_key=api_key)

# EasyOCR reader
try:
    logger.info("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'])
    logger.info("EasyOCR reader initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {str(e)}")
    raise RuntimeError(f"EasyOCR initialization failed: {str(e)}")

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extract text from image using EasyOCR"""
    try:
        logger.info("Running OCR...")
        result = reader.readtext(image_bytes, detail=0)
        extracted_text = ' '.join(result)
        logger.info(f"OCR completed. Extracted {len(extracted_text)} characters")
        return extracted_text
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")

def call_gemini_ai(extracted_text: str, language: str) -> str:
    """Send extracted text to Google Gemini for summarization"""
    # Validate language input
    allowed_languages = ["english", "hindi", "spanish", "french", "german", "chinese", "japanese", "hi", "en", "es", "fr", "de", "zh", "ja"]
    if language.lower() not in allowed_languages:
        logger.warning(f"Unsupported language requested: {language}")
        language = "english"
    
    # Map language codes to full names
    language_map = {
        "hi": "hindi", "en": "english", "es": "spanish", 
        "fr": "french", "de": "german", "zh": "chinese", "ja": "japanese"
    }
    output_language = language_map.get(language.lower(), language)

    prompt = f"""
    You are a multilingual medical assistant. Analyze the following medical report
    and respond in **{output_language}**.

    Provide:
    1. Summary of key findings
    2. Precautions
    3. Follow-up recommendations
    4. Ayurvedic remedies if applicable

    Important: Include disclaimer - "Consult with a qualified healthcare professional before making any medical decisions."

    Medical Report Text:
    {extracted_text}
    """

    try:
        logger.info(f"Calling Gemini AI with language: {output_language}")
        
        # Correct way to call Gemini API
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # You can also try "gemini-1.5-flash"
            contents=prompt,
        )

        # Correct way to extract response text
        if response and hasattr(response, 'text'):
            ai_response = response.text
            logger.info("Gemini AI response received successfully")
            return ai_response
        else:
            logger.error("Invalid response format from Gemini")
            # Debug: log the response structure
            logger.error(f"Response type: {type(response)}")
            logger.error(f"Response attributes: {dir(response)}")
            raise HTTPException(status_code=500, detail="AI processing failed - invalid response format")

    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")

@app.post("/process-medical-report")
async def process_medical_report(
    image: UploadFile = File(..., description="Medical report image file"),
    language: str = Form(..., description="Output language for summary")
):
    """Process medical report in English OCR, summarize in chosen language"""
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG, etc.)"
            )

        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024
        image_bytes = await image.read()
        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Processing report, target language: {language}")
        
        # Extract text from image
        extracted_text = extract_text_from_image(image_bytes)

        if not extracted_text.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No text extracted from image. Please ensure the image is clear, readable, and contains visible text."
                }
            )

        # Get AI summary
        ai_summary = call_gemini_ai(extracted_text, language)

        return {
            "success": True,
            "language": language,
            "extracted_text": extracted_text,
            "ai_summary": ai_summary,
            "image_base64": base64.b64encode(image_bytes).decode('utf-8'),
            "message": "Medical report processed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_medical_report: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during medical report processing"
        )

@app.get("/")
async def root():
    return {"message": "Medical Report Summarizer API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)