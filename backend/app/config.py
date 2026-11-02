"""\nConfiguration management\nLoads all configuration from the .env file in the project root\n"""

import os
from dotenv import load_dotenv

# Load the .env file from the project root
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If there is no .env in the root, try loading from environment variables (for production)
    load_dotenv(override=True)


class Config:
    """Flask configuration class"""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # JSON configuration - disable ASCII escaping so Chinese is displayed directly
    JSON_AS_ASCII = False
    
    # LLM configuration (uses the OpenAI format throughout)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    # "bedrock" routes calls through the Bedrock Converse adapter; empty means OpenAI-compatible
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', '')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
    
    # Zep configuration
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    # Write agent actions into the Zep graph during simulations. Off skips the
    # slowest Zep step; reports then read actions from the local action logs.
    ZEP_SIMULATION_MEMORY = os.environ.get('ZEP_SIMULATION_MEMORY', 'true').strip().lower() not in {
        'false', '0', 'no', 'off'
    }
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # Text processing configuration
    # Each chunk is one Zep episode processed at a few seconds apiece, so larger
    # chunks build the graph much faster and give extraction more context
    DEFAULT_CHUNK_SIZE = 2000
    DEFAULT_CHUNK_OVERLAP = 200
    
    # OASIS simulation configuration
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # Available actions per OASIS platform
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent configuration
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate required configuration"""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY is not configured")
        provider = cls.LLM_PROVIDER.strip().lower()
        if provider not in ("", "openai", "bedrock"):
            errors.append(f"LLM_PROVIDER must be empty, 'openai' or 'bedrock', got '{cls.LLM_PROVIDER}'")
        if provider == "bedrock":
            try:
                import botocore
                major, minor = (int(part) for part in botocore.__version__.split(".")[:2])
                if (major, minor) < (1, 39):
                    errors.append(f"LLM_PROVIDER=bedrock needs botocore>=1.39, found {botocore.__version__}")
            except ImportError:
                errors.append("LLM_PROVIDER=bedrock needs boto3; run `uv sync` in backend/")
        if not cls.ZEP_API_KEY:
            errors.append("ZEP_API_KEY is not configured")
        if os.environ.get("ZEP_API_URL"):
            errors.append("ZEP_API_URL is not supported; MiroFish only connects to Zep Cloud")
        if cls.DEBUG:
            import warnings
            warnings.warn("Flask DEBUG mode is enabled. Do not use in production.", RuntimeWarning)
        return errors
