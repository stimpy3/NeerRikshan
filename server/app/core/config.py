from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env against this file, not the current working directory. Otherwise
# running the pipeline from the repo root silently yields an empty Settings and
# a confusing "no project found" error from Earth Engine.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    database_url: str = ""

    # Earth Engine. gee_project is required by ee.Initialize; the service-account
    # pair is optional and only used for non-interactive runs. Interactive
    # `earthengine authenticate` credentials are the default path.
    gee_project: str = ""
    gee_service_account_email: str = ""
    gee_service_account_key_path: str = ""

    # Set only on AWS (Lambda env vars). Empty means read from server/data on
    # disk, which is what local development uses. See app.core.storage.
    data_bucket: str = ""
    site_bucket: str = ""


settings = Settings()
