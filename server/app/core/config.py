from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    database_url: str = ""

    gee_service_account_email: str = ""
    gee_service_account_key_path: str = ""


settings = Settings()
