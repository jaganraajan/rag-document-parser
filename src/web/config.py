import os

class BaseConfig:
    DEBUG = False
    TESTING = False
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

class DevConfig(BaseConfig):
    DEBUG = True

class ProdConfig(BaseConfig):
    pass

def select_config():
    env = os.getenv("APP_ENV", "dev").lower()
    if env.startswith("prod"):
        return ProdConfig
    return DevConfig