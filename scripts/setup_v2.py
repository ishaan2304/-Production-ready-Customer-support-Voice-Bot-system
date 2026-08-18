"""
VoiceBot v2 Setup Script.
Run this once to install dependencies and initialize the knowledge base.
"""
import subprocess
import sys
import json
from pathlib import Path


def run(cmd: str):
    print(f"\n► {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  Command failed (exit {result.returncode})")
    return result.returncode == 0


def main():
    print("=" * 60)
    print("VoiceBot v2 Setup")
    print("=" * 60)

    # Install dependencies
    print("\n[1/3] Installing v2 dependencies...")
    run(f"{sys.executable} -m pip install -r requirements_v2.txt")

    # Create .env if not exists
    env_path = Path(".env")
    if not env_path.exists():
        print("\n[2/3] Creating .env template...")
        env_path.write_text(
            "OPENAI_API_KEY=your_openai_api_key_here\n"
            "ELEVENLABS_API_KEY=your_elevenlabs_api_key_here\n"
            "ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM\n"
        )
        print("✅ .env created — add your API keys to this file!")
    else:
        print("\n[2/3] .env already exists — skipping")

    # Verify knowledge base
    print("\n[3/3] Verifying knowledge base...")
    kb_path = Path("data/knowledge_base.json")
    if kb_path.exists():
        with open(kb_path) as f:
            kb = json.load(f)
        print(f"✅ Knowledge base: {len(kb)} articles ready")
    else:
        print("⚠️  knowledge_base.json not found in data/")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add your API keys to .env")
    print("2. Start the server:")
    print("   uvicorn app.main_v2:app --host 0.0.0.0 --port 8000 --reload")
    print("3. Open: http://localhost:8000/docs")
    print("\nNote: RAG/ChromaDB index builds on first request (~30 seconds)")
    print("      ElevenLabs and GPT-4.1-mini need valid API keys in .env")
    print("      System gracefully falls back to gTTS + templates if keys missing")


if __name__ == "__main__":
    main()
