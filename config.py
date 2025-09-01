# --- ИСПРАВЛЕННАЯ ВЕРСИЯ config.py ---

import os
from dotenv import load_dotenv

# Эта строчка автоматически найдет твой .env файл и загрузит из него переменные.
# Это позволяет нам не хранить секреты прямо в коде.
load_dotenv()

# --- СЕКРЕТЫ И КЛЮЧИ (безопасно загружаются из .env) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Секреты для облачной памяти
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") 
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
# СТРОКА НИЖЕ БЫЛА УДАЛЕНА, ТАК КАК ТОКЕН ТЕПЕРЬ ЯВЛЯЕТСЯ ЧАСТЬЮ URL
# UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")


# --- НАСТРОЙКИ МОДЕЛЕЙ ---
# Словарь с доступными моделями. Ключ - то, что пишет пользователь, значение - реальное имя модели для API.
AVAILABLE_MODELS = {
    "pro": "gemini-1.5-pro-latest",
    "flash": "gemini-1.5-flash-latest",
}
# Модель по умолчанию, если пользователь не выбрал другую.
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-1.5-pro-latest")
# Модель для быстрых задач, таких как роутер.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gemini-1.5-flash-latest")


# --- НАСТРОЙКИ ПАМЯТИ И ЛИМИТОВ ---
# int() преобразует текст из .env в число. Если переменной нет, используется значение по умолчанию (например, 20).
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 20))
LTM_SLOT_LIMIT = int(os.getenv("LTM_SLOT_LIMIT", 500))
MAX_FILE_SIZE_BYTE = int(os.getenv("MAX_FILE_SIZE_BYTE", 102400)) # 100 КБ

# --- ПРОВЕРКА КРИТИЧЕСКИХ ПЕРЕМЕННЫХ ---
# Если основные ключи не найдены, бот не сможет запуститься, и мы увидим понятную ошибку.
if not all([TELEGRAM_BOT_TOKEN, GOOGLE_API_KEY, PINECONE_API_KEY, UPSTASH_REDIS_REST_URL]):
    raise ValueError("ОДНА ИЛИ НЕСКОЛЬКО КРИТИЧЕСКИ ВАЖНЫХ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ НЕ УСТАНОВЛЕНЫ!")