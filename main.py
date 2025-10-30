import os
import re
import json
import math
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pypdf import PdfReader
try:
    import tiktoken
except Exception:
    tiktoken = None

# -------------------------
# Config & Globals
# -------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "120000"))
RESERVED_COMPLETION_TOKENS = int(os.getenv("RESERVED_COMPLETION_TOKENS", "2000"))
SYSTEM_TOKENS_BUDGET = int(os.getenv("SYSTEM_TOKENS_BUDGET", "800"))
KEYWORD_PRIORITY = True
KEYWORD_WINDOW = 6
DOC_STORE: Dict[str, Any] = {"pages": [], "path": None, "loaded_at": None}
EMBEDDED_PDF_PATH = "catalog.pdf"

# -------------------------
# Models
# -------------------------
class AskInput(BaseModel):
    question: str
    max_context_tokens: Optional[int] = None

class AskResponse(BaseModel):
    answer: str

# -------------------------
# PDF → Text
# -------------------------
def load_pdf(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = normalize_text(txt)
        pages.append({"page_num": i + 1, "text": txt})
    return {"path": path, "pages": pages, "loaded_at": int(time.time())}

def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\u00ad", "", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

# -------------------------
# Token Estimation
# -------------------------
def estimate_tokens(text: str) -> int:
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / 4))

# -------------------------
# Context Packing
# -------------------------
def slice_keyword_excerpts(text: str, keywords: List[str], window: int = KEYWORD_WINDOW) -> List[str]:
    if not keywords:
        return []
    tokens = re.findall(r"\w+|\W+", text)
    lower_tokens = [t.lower() for t in tokens]
    joined = "".join(tokens)
    words = re.findall(r"\w+", text)
    positions: List[Tuple[int, int]] = []
    w_lower = [w.lower() for w in words]
    for kw in keywords:
        kw_l = kw.lower()
        for i, w in enumerate(w_lower):
            if w == kw_l:
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                positions.append((start, end))
    positions.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in positions:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    excerpts = []
    word_iter = re.finditer(r"\w+", text)
    word_spans = [m.span() for m in word_iter]
    for s, e in merged:
        if s >= len(word_spans):
            continue
        start_char = word_spans[s][0]
        end_char = word_spans[min(e - 1, len(word_spans) - 1)][1]
        excerpts.append(text[start_char:end_char])
    return excerpts

def pack_context(question: str, pages: List[Dict[str, Any]], max_context_tokens: int) -> str:
    keywords = re.findall(r"[A-Za-z0-9]{3,}", question)
    header = ""
    budget = max(1000, max_context_tokens - estimate_tokens(header))
    used_tokens = 0
    parts: List[str] = []

    def add_block(text: str) -> bool:
        nonlocal used_tokens
        block = f"\n\n{text.strip()}\n"
        t = estimate_tokens(block)
        if used_tokens + t > budget:
            return False
        parts.append(block)
        used_tokens += t
        return True

    if KEYWORD_PRIORITY and keywords:
        for p in pages:
            excerpts = slice_keyword_excerpts(p["text"], keywords)
            if not excerpts:
                continue
            excerpt = " ... \n".join(excerpts[:3])
            if not excerpt.strip():
                continue
            ok = add_block(excerpt)
            if not ok:
                break
    if used_tokens < budget:
        for p in pages:
            ok = add_block(p["text"])
            if not ok:
                break
    context = header + "\n".join(parts)
    return context.strip()

# -------------------------
# URL Cleaning
# -------------------------
def clean_urls_in_text(text: str) -> str:
    def clean_url(match):
        url = match.group(1)
        url = url.rstrip('.,;!?')
        return url
    pattern = r'(\bhttps?://\S+[^\s.,;!?])[.,;!?]?'
    return re.sub(pattern, clean_url, text)

# -------------------------
# Prompt Templates
# -------------------------
SYSTEM_PROMPT = (
    "You are a virtual engineering library assistant named Libro. "
    "Your primary knowledge base is the provided CONTEXT, which contains engineering-related information. "
    "Follow these rules strictly:\n"
    "1. If the user greets you (e.g., 'hello', 'hi', 'good morning'), respond with a friendly greeting and ask how you can help with engineering topics.\n"
    "2. If the answer is in the CONTEXT, use it to answer the question.\n"
    "3. If the answer is NOT in the CONTEXT but the question is related to engineering, use your general knowledge to provide a helpful and accurate response.\n"
    "4. If the question is NOT related to engineering, respond with: "
    "'I specialize in engineering topics. How can I assist you with engineering-related questions?'\n"
    "5. Never mention the CONTEXT, provided text, or page numbers. "
    "6. Always respond in a natural, professional, and conversational tone.\n"
    "7. If your answer contains a URL, do not add any punctuation (like a period or comma) immediately after it."
)

ANSWER_PROMPT_TMPL = (
    "USER QUESTION:\n{question}\n\n"
    "RELEVANT INFORMATION FROM ENGINEERING DOCUMENTS:\n{context}\n\n"
    "INSTRUCTIONS:\n"
    "- If the user is greeting you, respond with a friendly greeting and ask how you can help with engineering topics.\n"
    "- First, check if the answer is in the RELEVANT INFORMATION. If yes, use it.\n"
    "- If the answer is NOT in the RELEVANT INFORMATION but the question is about engineering, use your general knowledge to answer.\n"
    "- If the question is NOT about engineering, respond with: "
    "'I specialize in engineering topics. How can I assist you with engineering-related questions?'\n"
    "- Never mention the RELEVANT INFORMATION, CONTEXT, or page numbers.\n"
    "- Respond naturally and professionally.\n"
    "- If your answer contains a URL, do not add any punctuation (like a period or comma) immediately after it."
)

# -------------------------
# OpenRouter Chat Call (using OpenAI client)
# -------------------------
def openrouter_chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 1500) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    completion = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://your-site-url.com",  # Replace with your site URL
            "X-Title": "Virtual Engineering Library Assistant",  # Replace with your site name
        },
        model="openai/gpt-oss-20b:free",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()

# -------------------------
# Orchestration
# -------------------------
def build_answer(question: str, model_ctx_tokens: int) -> Dict[str, Any]:
    if not DOC_STORE.get("pages"):
        raise HTTPException(status_code=400, detail="No PDF loaded")
    budget = max(2000, model_ctx_tokens - RESERVED_COMPLETION_TOKENS - SYSTEM_TOKENS_BUDGET)
    context = pack_context(question, DOC_STORE["pages"], budget)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ANSWER_PROMPT_TMPL.format(question=question, context=context)},
    ]
    max_out = min(RESERVED_COMPLETION_TOKENS, 4000)
    content = openrouter_chat(messages, temperature=0.2, max_tokens=max_out)
    content = clean_urls_in_text(content)
    return {
        "answer": content,
        "doc_path": DOC_STORE.get("path"),
        "pages_used": len(DOC_STORE.get("pages", [])),
        "context_tokens": estimate_tokens(context),
    }

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="Virtual Engineering Library Assistant")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only! Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global DOC_STORE
    DOC_STORE = load_pdf(EMBEDDED_PDF_PATH)

@app.post("/ask", response_model=AskResponse)
def api_ask(body: AskInput):
    try:
        ctx_tokens = body.max_context_tokens or MAX_CONTEXT_TOKENS
        result = build_answer(body.question, ctx_tokens)
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Local Test
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
