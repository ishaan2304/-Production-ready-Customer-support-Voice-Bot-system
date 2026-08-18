"""
VoiceBot v2 — Upgraded FastAPI application.
Integrates GPT-4.1-mini, RAG (ChromaDB), ElevenLabs TTS, and WebSocket streaming.
"""
import json
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from app.config import get_config
from app.logger import setup_logging, get_logger, set_request_id
from app.schemas import (
    TranscribeResponse, IntentResponse, IntentPrediction,
    ResponseGenerateRequest, ResponseGenerateResponse,
    SynthesizeRequest, HealthResponse, ErrorResponse, TextRequest,
)
from app.exceptions import VoiceBotError

config = get_config()
setup_logging(
    log_level=config.app.get("log_level", "INFO"),
    log_file=config.app.get("log_file", "logs/voicebot.log"),
)
logger = get_logger(__name__)

# Module singletons
_asr = _classifier = _rag = _llm = _tts = None
_start_time = time.time()


def get_asr():
    global _asr
    if _asr is None:
        from app.asr import ASRModule
        _asr = ASRModule()
    return _asr

def get_classifier():
    global _classifier
    if _classifier is None:
        from app.intent_classifier import IntentClassifier
        _classifier = IntentClassifier()
    return _classifier

def get_rag():
    global _rag
    if _rag is None:
        from app.rag import RAGModule
        _rag = RAGModule()
    return _rag

def get_llm():
    global _llm
    if _llm is None:
        from app.llm import LLMGenerator
        _llm = LLMGenerator()
    return _llm

def get_tts():
    global _tts
    if _tts is None:
        from app.tts_elevenlabs import ElevenLabsTTS
        _tts = ElevenLabsTTS()
    return _tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"Starting VoiceBot v2 — GPT-4.1-mini + RAG + ElevenLabs")
    logger.info("=" * 60)
    try:
        get_classifier()._lazy_load()
        get_llm()._lazy_load()
        get_tts()._lazy_load()
        logger.info("Core modules initialized")
    except Exception as e:
        logger.warning(f"Startup warning: {e}")
    yield
    logger.info("VoiceBot shutting down")


app = FastAPI(
    title="VoiceBot v2 — AI Customer Support",
    description="Voice Bot with GPT-4.1-mini, RAG (ChromaDB), ElevenLabs TTS, and WebSocket streaming.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    rid = set_request_id()
    request.state.request_id = rid
    request.state.start_time = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    elapsed = (time.perf_counter() - request.state.start_time) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
    return response


@app.exception_handler(VoiceBotError)
async def voicebot_error_handler(request: Request, exc: VoiceBotError):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.message, details=exc.details).model_dump(),
    )


# ─── HEALTH ───────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    llm = get_llm()
    tts = get_tts()
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        models_loaded={
            "asr": get_asr().is_loaded,
            "intent_classifier": get_classifier().is_loaded,
            "llm_gpt4_mini": llm.is_loaded,
            "rag_chromadb": get_rag().is_loaded,
            "tts_elevenlabs": tts.is_loaded,
            "tts_engine": tts.engine_name,
            "llm_fallback": llm.using_fallback,
        },
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ─── ASR ──────────────────────────────────────────────────────


@app.post("/transcribe", response_model=TranscribeResponse, tags=["ASR"])
async def transcribe_audio(request: Request, audio: UploadFile = File(...)):
    """Transcribe WAV audio to text using Whisper."""
    request_id = getattr(request.state, "request_id", set_request_id())
    if not audio.filename.lower().endswith(".wav"):
        from app.exceptions import AudioInputError
        raise AudioInputError("Only WAV format supported")
    audio_bytes = await audio.read()
    result = get_asr().transcribe(audio_bytes)
    return TranscribeResponse(
        transcript=result["transcript"],
        language=result["language"],
        confidence=result["confidence"],
        duration_seconds=result["duration_seconds"],
        request_id=request_id,
        processing_time_ms=result["processing_time_ms"],
    )


# ─── INTENT ───────────────────────────────────────────────────


@app.post("/predict-intent", response_model=IntentResponse, tags=["NLP"])
async def predict_intent(request: Request, body: TextRequest):
    """Classify user intent from text using fine-tuned DistilBERT."""
    request_id = getattr(request.state, "request_id", set_request_id())
    result = get_classifier().predict(body.text)
    top = result["top_intent"]
    return IntentResponse(
        text=body.text,
        top_intent=IntentPrediction(**top),
        all_intents=[IntentPrediction(**i) for i in result["all_intents"]],
        is_confident=result["is_confident"],
        request_id=request_id,
        processing_time_ms=result["processing_time_ms"],
    )


# ─── RAG ──────────────────────────────────────────────────────


@app.post("/retrieve-context", tags=["RAG"])
async def retrieve_context(body: TextRequest):
    """Retrieve relevant knowledge base articles for a query."""
    rag = get_rag()
    docs = rag.retrieve(body.text)
    return {
        "query": body.text,
        "documents": docs,
        "num_retrieved": len(docs),
        "context": rag.format_context(docs),
    }


# ─── LLM RESPONSE ─────────────────────────────────────────────


@app.post("/generate-response", tags=["NLP"])
async def generate_response(request: Request, body: ResponseGenerateRequest):
    """
    Generate AI response using GPT-4.1-mini + RAG context.
    Retrieves relevant knowledge base articles before generating.
    """
    request_id = getattr(request.state, "request_id", set_request_id())
    start = time.perf_counter()

    # Get intent
    if body.intent:
        intent = body.intent
        confidence = 0.9
    else:
        clf_result = get_classifier().predict(body.text)
        intent = clf_result["top_intent"]["intent"]
        confidence = clf_result["top_intent"]["confidence"]

    # RAG retrieval
    rag = get_rag()
    try:
        docs = rag.retrieve(body.text)
        context = rag.format_context(docs)
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        docs = []
        context = ""

    # LLM generation
    llm = get_llm()
    result = llm.generate(
        user_text=body.text,
        intent=intent,
        rag_context=context,
        conversation_history=body.context.get("history") if body.context else None,
    )

    elapsed = (time.perf_counter() - start) * 1000
    return {
        "response_text": result["response_text"],
        "intent_used": intent,
        "model_used": result.get("model_used", "unknown"),
        "rag_documents_used": len(docs),
        "tokens_used": result.get("tokens_used", 0),
        "request_id": request_id,
        "processing_time_ms": round(elapsed, 2),
    }


# ─── TTS ──────────────────────────────────────────────────────


@app.post("/synthesize", tags=["TTS"])
async def synthesize_speech(body: SynthesizeRequest):
    """Convert text to speech using ElevenLabs or gTTS fallback."""
    tts = get_tts()
    result = tts.synthesize(text=body.text, language=body.language, slow=body.slow)
    return Response(
        content=result["audio_bytes"],
        media_type="audio/mpeg",
        headers={
            "X-TTS-Engine": result["engine"],
            "X-Duration-Estimate": str(result["duration_estimate_seconds"]),
            "X-Processing-Time-Ms": str(result["processing_time_ms"]),
            "Content-Disposition": 'attachment; filename="response.mp3"',
        },
    )


@app.get("/voices", tags=["TTS"])
async def list_voices():
    """List available ElevenLabs voices."""
    return {"voices": get_tts().list_voices(), "engine": get_tts().engine_name}


# ─── UNIFIED VOICEBOT ─────────────────────────────────────────


@app.post("/voicebot", tags=["VoiceBot"])
async def voicebot_pipeline(
    request: Request,
    audio: UploadFile = File(...),
    return_metadata: bool = Form(default=False),
):
    """
    Unified end-to-end pipeline: WAV Audio → MP3 Audio
    ASR → Intent → RAG → GPT-4.1-mini → ElevenLabs TTS
    """
    request_id = getattr(request.state, "request_id", set_request_id())
    pipeline_start = time.perf_counter()
    timings = {}

    if not audio.filename.lower().endswith(".wav"):
        from app.exceptions import AudioInputError
        raise AudioInputError("Only WAV audio supported")

    audio_bytes = await audio.read()
    logger.info(f"[{request_id}] VoiceBot v2 pipeline started ({len(audio_bytes)} bytes)")

    # Step 1: ASR
    t0 = time.perf_counter()
    asr_result = get_asr().transcribe(audio_bytes)
    timings["asr_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    transcript = asr_result["transcript"]

    # Step 2: Intent
    t0 = time.perf_counter()
    intent_result = get_classifier().predict(transcript)
    timings["intent_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    top_intent = intent_result["top_intent"]

    # Step 3: RAG
    t0 = time.perf_counter()
    try:
        docs = get_rag().retrieve(transcript)
        context = get_rag().format_context(docs)
    except Exception:
        docs, context = [], ""
    timings["rag_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # Step 4: LLM
    t0 = time.perf_counter()
    llm_result = get_llm().generate(
        user_text=transcript,
        intent=top_intent["intent"],
        rag_context=context,
    )
    timings["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    response_text = llm_result["response_text"]

    # Step 5: TTS
    t0 = time.perf_counter()
    tts_result = get_tts().synthesize(text=response_text)
    timings["tts_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    total = round((time.perf_counter() - pipeline_start) * 1000, 2)
    logger.info(f"[{request_id}] Pipeline complete in {total}ms | {timings}")

    return Response(
        content=tts_result["audio_bytes"],
        media_type="audio/mpeg",
        headers={
            "X-Request-ID": request_id,
            "X-Transcript": transcript[:200],
            "X-Intent": top_intent["intent"],
            "X-Intent-Confidence": str(round(top_intent["confidence"], 4)),
            "X-Response-Text": response_text[:200],
            "X-Model-Used": llm_result.get("model_used", "unknown"),
            "X-RAG-Docs": str(len(docs)),
            "X-TTS-Engine": tts_result["engine"],
            "X-Total-Latency-Ms": str(total),
            "X-ASR-Ms": str(timings["asr_ms"]),
            "X-Intent-Ms": str(timings["intent_ms"]),
            "X-RAG-Ms": str(timings["rag_ms"]),
            "X-LLM-Ms": str(timings["llm_ms"]),
            "X-TTS-Ms": str(timings["tts_ms"]),
            "Content-Disposition": 'attachment; filename="voicebot_response.mp3"',
        },
    )


# ─── WEBSOCKET STREAMING ──────────────────────────────────────


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming responses.
    Client sends text, server streams GPT response token by token.

    Message format from client: {"text": "user query"}
    Server streams: {"type": "token", "content": "word"}
    Server sends:   {"type": "done", "intent": "...", "full_response": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            user_text = message.get("text", "").strip()

            if not user_text:
                await websocket.send_json({"type": "error", "content": "Empty message"})
                continue

            logger.info(f"WS received: '{user_text[:60]}'")

            # Classify intent
            await websocket.send_json({"type": "status", "content": "Classifying intent..."})
            intent_result = get_classifier().predict(user_text)
            top_intent = intent_result["top_intent"]

            await websocket.send_json({
                "type": "intent",
                "intent": top_intent["intent"],
                "confidence": top_intent["confidence"],
            })

            # RAG retrieval
            await websocket.send_json({"type": "status", "content": "Searching knowledge base..."})
            try:
                docs = get_rag().retrieve(user_text)
                context = get_rag().format_context(docs)
            except Exception:
                docs, context = [], ""

            await websocket.send_json({
                "type": "rag",
                "num_docs": len(docs),
            })

            # Stream LLM response
            await websocket.send_json({"type": "status", "content": "Generating response..."})
            full_response = ""

            for token in get_llm().generate_stream(
                user_text=user_text,
                intent=top_intent["intent"],
                rag_context=context,
            ):
                full_response += token
                await websocket.send_json({"type": "token", "content": token})

            # Done
            await websocket.send_json({
                "type": "done",
                "intent": top_intent["intent"],
                "full_response": full_response,
                "rag_docs_used": len(docs),
            })

            logger.info(f"WS response streamed: {len(full_response)} chars")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main_v2:app", host="0.0.0.0", port=8000, reload=True)
