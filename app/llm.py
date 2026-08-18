"""
LLM Response Generation Module.
Uses GPT-4.1-mini with RAG context for intelligent,
grounded customer support responses.
Supports both standard and streaming generation.
"""
import os
import time
from typing import Dict, Any, Optional, Generator

from app.config import get_config
from app.exceptions import ResponseGenerationError
from app.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful, professional customer support agent for an e-commerce company.

Your role:
- Answer customer questions accurately and empathetically
- Use the provided knowledge base context to give accurate information
- Be concise but complete — aim for 2-4 sentences
- Always maintain a friendly, professional tone
- If you don't know something, say so honestly and offer to escalate
- Never make up information not in the context
- Stay strictly within customer support scope

Important rules:
- Do NOT discuss topics unrelated to customer support
- Do NOT make promises you cannot keep
- Always refer to specific policies when available
- End responses with a helpful follow-up offer when appropriate
"""


class LLMGenerator:
    """
    GPT-4.1-mini powered response generator with RAG integration.
    Falls back to template-based generation if OpenAI is unavailable.
    """

    def __init__(self):
        self.config = get_config()
        self._client = None
        self._loaded = False
        self._use_fallback = False

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        try:
            from openai import OpenAI
            from dotenv import load_dotenv
            load_dotenv()

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your_openai_api_key_here":
                logger.warning("OpenAI API key not set — using template fallback")
                self._use_fallback = True
                self._loaded = True
                return

            self._client = OpenAI(api_key=api_key)
            self._loaded = True
            self._use_fallback = False
            logger.info("LLM generator ready (GPT-4.1-mini)")

        except ImportError:
            logger.warning("openai package not installed — using template fallback")
            self._use_fallback = True
            self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def using_fallback(self) -> bool:
        return self._use_fallback

    def generate(
        self,
        user_text: str,
        intent: str,
        rag_context: str = "",
        conversation_history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response using GPT-4.1-mini + RAG context.

        Args:
            user_text: Original user query
            intent: Classified intent label
            rag_context: Retrieved knowledge base context
            conversation_history: Previous turns for multi-turn support

        Returns:
            Dict with response_text, model_used, tokens_used
        """
        self._lazy_load()
        start = time.perf_counter()

        if self._use_fallback:
            return self._fallback_generate(user_text, intent)

        try:
            messages = self._build_messages(
                user_text, intent, rag_context, conversation_history
            )

            response = self._client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.4,
                presence_penalty=0.1,
            )

            response_text = response.choices[0].message.content.strip()
            elapsed = (time.perf_counter() - start) * 1000

            logger.info(
                f"LLM generated response in {elapsed:.1f}ms "
                f"(tokens: {response.usage.total_tokens})"
            )

            return {
                "response_text": response_text,
                "model_used": "gpt-4.1-mini",
                "tokens_used": response.usage.total_tokens,
                "processing_time_ms": round(elapsed, 2),
                "rag_used": bool(rag_context),
            }

        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            logger.warning("Falling back to template response")
            return self._fallback_generate(user_text, intent)

    def generate_stream(
        self,
        user_text: str,
        intent: str,
        rag_context: str = "",
        conversation_history: Optional[list] = None,
    ) -> Generator[str, None, None]:
        """
        Stream response tokens as they are generated.
        Yields text chunks for WebSocket streaming.
        """
        self._lazy_load()

        if self._use_fallback:
            result = self._fallback_generate(user_text, intent)
            # Simulate streaming for fallback
            words = result["response_text"].split()
            for i, word in enumerate(words):
                yield word + (" " if i < len(words) - 1 else "")
                time.sleep(0.05)
            return

        try:
            messages = self._build_messages(
                user_text, intent, rag_context, conversation_history
            )

            stream = self._client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.4,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield "I apologize, I'm having trouble generating a response right now. Please try again."

    def _build_messages(
        self,
        user_text: str,
        intent: str,
        rag_context: str,
        history: Optional[list],
    ) -> list:
        """Build the messages array for the OpenAI API."""
        intent_display = intent.replace("_", " ").title()

        user_message = f"""Customer Query: {user_text}

Detected Intent: {intent_display}

Relevant Knowledge Base Context:
{rag_context if rag_context else 'No specific articles found.'}

Please provide a helpful, accurate response based on the context above."""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history for multi-turn
        if history:
            for turn in history[-6:]:  # Last 3 turns
                messages.append(turn)

        messages.append({"role": "user", "content": user_message})
        return messages

    def _fallback_generate(self, user_text: str, intent: str) -> Dict[str, Any]:
        """Fall back to template-based generation."""
        from app.response_generator import ResponseGenerator
        gen = ResponseGenerator()
        result = gen.generate(intent=intent, confidence=0.9)
        result["model_used"] = "template_fallback"
        result["tokens_used"] = 0
        result["rag_used"] = False
        return result
