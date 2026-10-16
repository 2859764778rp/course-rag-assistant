import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path, override=False)
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          
PHASE2_DIR = os.path.join(BASE_DIR, "phase2")
DATA_DIR = os.path.join(PHASE2_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")         
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")    
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")    
SQLITE_PATH = os.path.join(DATA_DIR, "meta.db") 
DOCS_DIR = os.path.join(PHASE2_DIR, "docs")

for _d in (RAW_DIR, CORPUS_DIR, CHROMA_DIR, DOCS_DIR):
    os.makedirs(_d, exist_ok=True)

_LLM_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
if not _LLM_KEY:
    raise RuntimeError(
        "未检测到 API key：请在项目根目录 .env 中配置 DEEPSEEK_API_KEY 后重试。"
    )
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = _LLM_KEY
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 60

EMB_BACKEND = os.environ.get("EMB_BACKEND", "local")

EMB_OPENAI_MODEL = os.environ.get("EMB_OPENAI_MODEL", "text-embedding-3-small")
EMB_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
EMB_OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

EMB_LOCAL_MODEL = os.environ.get("EMB_LOCAL_MODEL", "BAAI/bge-large-zh-v1.5")
EMB_LOCAL_FALLBACK = os.environ.get("EMB_LOCAL_FALLBACK", "BAAI/bge-large-zh-v1.5")


CHUNK_SIZE = 600      
CHUNK_OVERLAP = 100   


RETRIEVER_TOP_K = 8      
HYBRID_ENABLE = False  

RAG_SEARCH_FALLBACK = os.environ.get("RAG_SEARCH_FALLBACK", "true").strip().lower() in ("1", "true", "yes")

RAG_FALLBACK_DISTANCE = float(os.environ.get("RAG_FALLBACK_DISTANCE", "0.95"))


INTENT_CONFIDENCE_THRESHOLD = 0.6  
INTENT_LLM_TIMEOUT = 30

SEARCH_BACKEND = os.environ.get("SEARCH_BACKEND", "auto").strip().lower()
BOCHA_API_KEY = os.environ.get("BOCHA_API_KEY", "")
if SEARCH_BACKEND == "bocha" and not BOCHA_API_KEY:
    raise RuntimeError(
        "SEARCH_BACKEND=bocha 但未配置 BOCHA_API_KEY：请在项目根目录 .env 中配置后重试。"
    )
BOCHA_SEARCH_URL = os.environ.get("BOCHA_SEARCH_URL", "https://api.bochaai.com/v1/web-search")
BOCHA_SEARCH_TIMEOUT = int(os.environ.get("BOCHA_SEARCH_TIMEOUT", "30"))
BOCHA_SEARCH_COUNT = int(os.environ.get("BOCHA_SEARCH_COUNT", "5"))

AUTO_SEARCH_TIMEOUT = int(os.environ.get("AUTO_SEARCH_TIMEOUT", "8"))
AUTO_SEARCH_COOLDOWN = int(os.environ.get("AUTO_SEARCH_COOLDOWN", "60"))


OBJ_STORE_BACKEND = os.environ.get("OBJ_STORE_BACKEND", "local")
LOCAL_STORE_DIR = os.path.join(DATA_DIR, "objects")   
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "127.0.0.1:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "")
MINIO_BUCKET = "course-docs"
os.makedirs(LOCAL_STORE_DIR, exist_ok=True)

ALLOWED_EXTS = {"pdf", "docx", "txt", "pptx", "html", "ipynb"}
MAX_SIZE = 32 * 1024 * 1024  


HISTORY_WINDOW = int(os.environ.get("HISTORY_WINDOW", "5"))          
ROUTER_CONTEXT_TURNS = int(os.environ.get("ROUTER_CONTEXT_TURNS", "2")) 
QUESTION_RECOMMEND = os.environ.get("QUESTION_RECOMMEND", "true").strip().lower() in ("1", "true", "yes")
RECOMMEND_COUNT = int(os.environ.get("RECOMMEND_COUNT", "3"))        
