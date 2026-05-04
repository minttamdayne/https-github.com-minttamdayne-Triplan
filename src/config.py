from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file."""

    # ── Paths ──
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    # ── LLM (Đã chuyển mặc định sang Ollama để tránh lỗi 404) ──
    openai_api_key: str = ""
    llm_provider: str = "ollama"  # "ollama" | "openai"
    llm_model: str = "llama3.2"    # Đổi từ gpt-4o-mini sang llama3.2
    chat_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434/v1"
    
    # Thêm timeout để tránh treo pipeline khi chạy local LLM
    llm_timeout: float = 60.0 

    # ── Neo4j (Cập nhật mật khẩu bạn đã test OK) ──
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "Minhtam0402"

    # ── Web Search ──
    search_api_key: str = ""
    search_provider: str = "tavily"

    # ── Geocoding ──
    google_maps_api_key: str = ""

    # ── App ──
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Trip defaults ──
    default_daily_hours: float = 10.0
    max_candidates_per_day: int = 8

    # ── Scoring weights (Ưu tiên: Interest > Quality > Budget > Proximity) ──
    # Với tư cách là Researcher, bạn có thể tinh chỉnh các trọng số này để thử nghiệm
    w_interest: float = 0.40
    w_quality: float = 0.30
    w_budget: float = 0.20
    w_proximity: float = 0.10

    # ── Semantic & Festival settings ──
    # min_semantic_score removed — hard threshold replaced by soft penalty ranking.
    # Tune SOFT_PENALTY_FLOOR / SOFT_PENALTY_SCALE in semantic_agent.py instead.
    festival_scarcity_bonus: float = 0.15  # Additive bonus for festival POIs

    # ── Semantic selection bounds ──
    semantic_target_min: int = 500   # Minimum candidate pool size
    semantic_target_max: int = 800   # Maximum candidate pool size

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

settings = Settings()