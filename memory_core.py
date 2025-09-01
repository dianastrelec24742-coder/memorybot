
# Файл: memory_core.py (Версия для Render/Cloud с ленивой загрузкой)

import os
import json
import logging
from datetime import datetime
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
from redis.asyncio import Redis
from pinecone import Pinecone, PodSpec

# --- Инициализация ---
logger = logging.getLogger(__name__)

# --- ИЗМЕНЕНИЯ ДЛЯ ЛЕНИВОЙ ЗАГРУЗКИ ---

# 1. Глобальная переменная для модели теперь пустая. Модель НЕ загружается при старте.
MODEL = None 
# 2. Размерность вектора для модели 'all-MiniLM-L6-v2' известна заранее (384),
# поэтому мы ее просто указываем. Это избавляет от необходимости загружать модель на старте.
VECTOR_DIMENSION = 384

def get_model():
    """
    Функция-помощник для "ленивой" загрузки модели.
    При первом вызове она загружает модель в глобальную переменную MODEL.
    При всех последующих вызовах она просто возвращает уже загруженную модель.
    """
    global MODEL
    if MODEL is None:
        logger.info("Первый вызов. Загрузка модели SentenceTransformer в память...")
        # Эта строка выполнится только один раз, когда боту понадобится модель
        MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Модель SentenceTransformer успешно загружена.")
    return MODEL

# --- КОНЕЦ ИЗМЕНЕНИЙ ---


# --- Подключение к облачным сервисам (ИСПРАВЛЕННАЯ ВЕРСИЯ) ---
try:
    # Pinecone (без изменений)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    
    # --- 1. УДАЛЯЕМ лишнюю переменную для токена ---
    # Upstash Redis - теперь используем только одну полную URL
    UPSTASH_FULL_URL = os.getenv("UPSTASH_REDIS_REST_URL")

    # --- 2. УПРОЩАЕМ проверку переменных ---
    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, UPSTASH_FULL_URL]):
        raise ValueError("Одна или несколько переменных окружения для Pinecone/Upstash не установлены!")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        logger.warning(f"Индекс Pinecone '{PINECONE_INDEX_NAME}' не найден. Создаю новый...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="cosine",
            spec=PodSpec(environment=PINECONE_ENVIRONMENT)
        )
        logger.info(f"Индекс '{PINECONE_INDEX_NAME}' успешно создан.")
    
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # --- 3. ИСПРАВЛЯЕМ создание клиента Redis ---
    # Теперь передаем только полный URL.
    # Метод from_url сам извлечет пароль, хост, порт и включит SSL (потому что URL начинается с rediss://)
    redis_client = Redis.from_url(UPSTASH_FULL_URL, decode_responses=True, health_check_interval=30)
    


except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при инициализации облачных сервисов: {e}", exc_info=True)
    # exc_info=True добавлено для более детальных логов ошибки
    exit()

class MemoryCore:
    """
    Асинхронная версия для управления LTM с использованием Pinecone и Redis.
    """
    # <--- ИЗМЕНЕНО: `init` заменен на стандартный конструктор `init`
    def __init__(self, user_id, limit=1000):
        self.user_id = str(user_id)
        self.limit = limit
        self.redis_key_entries = f"ltm:{self.user_id}:entries"
        self.redis_key_queue = f"ltm:{self.user_id}:queue"
        self.redis_key_id_counter = f"ltm:{self.user_id}:next_id"

    @classmethod
    async def create(cls, user_id, limit=1000):
        instance = cls(user_id, limit)
        await instance._initialize_counter()
        return instance

    async def _initialize_counter(self):
        await redis_client.setnx(self.redis_key_id_counter, 0)

    async def _get_next_id(self):
        return await redis_client.incr(self.redis_key_id_counter)

    async def add_to_memory(self, text_chunk: str, is_eternal: bool = False, key: str = None) -> tuple[bool, int]:
        try:
            new_id = await self._get_next_id()
            
            if not is_eternal:
                queue_length = await redis_client.llen(self.redis_key_queue)
                if queue_length >= self.limit:
                    oldest_id_str = await redis_client.lpop(self.redis_key_queue)
                    if oldest_id_str:
                        await pinecone_index.delete(ids=[oldest_id_str], namespace=self.user_id)
                        await redis_client.hdel(self.redis_key_entries, oldest_id_str)
                        logger.info(f"Лимит достигнут. Удалено временное воспоминание с ID {oldest_id_str} для user {self.user_id}")
            
            # <--- ИЗМЕНЕНО: Вместо MODEL.encode вызываем get_model().encode
            vector = await asyncio.to_thread(get_model().encode, [text_chunk])

            entry_data = {
                "id": new_id,
                "timestamp": datetime.now().isoformat(),
                "content": text_chunk,
                "key": key,
                "type": "eternal_fact" if is_eternal else "contextual_memory"
            }
            
            await asyncio.to_thread(
                pinecone_index.upsert,vectors=[{"id": str(new_id), "values": vector[0].tolist()}],
                namespace=self.user_id
            )

            await redis_client.hset(self.redis_key_entries, str(new_id), json.dumps(entry_data))
            
            if not is_eternal:
                await redis_client.rpush(self.redis_key_queue, str(new_id))

            logger.info(f"Успешно добавлена запись ID {new_id} для user {self.user_id} (вечная: {is_eternal}).")
            return True, new_id

        except Exception as e:
            logger.error(f"Не удалось добавить запись в память для user {self.user_id}: {e}", exc_info=True)
            return False, -1

    async def search_in_memory(self, query_text: str, k: int = 5, distance_threshold: float = 0.3, search_type: str = "all") -> list:
        try:
            # <--- ИЗМЕНЕНО: Вместо MODEL.encode вызываем get_model().encode
            query_vector = await asyncio.to_thread(get_model().encode, [query_text])
            
            pinecone_filter = None
            if search_type == "eternal_only":
                k = k * 3

            search_response = await asyncio.to_thread(
                pinecone_index.query,
                vector=query_vector[0].tolist(),
                top_k=k,
                namespace=self.user_id,
                filter=pinecone_filter
            )

            match_ids = [match['id'] for match in search_response['matches'] if match['score'] >= (1 - distance_threshold)]
            if not match_ids:
                return []

            retrieved_entries_json = await redis_client.hmget(self.redis_key_entries, match_ids)
            
            relevant_memories = []
            for entry_json in retrieved_entries_json:
                if entry_json:
                    entry_data = json.loads(entry_json)
                    if search_type == "eternal_only" and entry_data.get("type") != "eternal_fact":
                        continue
                    relevant_memories.append(entry_data)

            relevant_memories.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return relevant_memories

        except Exception as e:
            logger.error(f"Ошибка при поиске в памяти для user {self.user_id}: {e}", exc_info=True)
            return []
            
    async def get_eternal_list(self) -> list:
        all_entries = await redis_client.hgetall(self.redis_key_entries)
        eternal_facts = []
        for entry_json in all_entries.values():
            entry_data = json.loads(entry_json)
            if entry_data.get("type") == "eternal_fact":
                eternal_facts.append(entry_data)
        return sorted(eternal_facts, key=lambda x: x['id'])
    async def get_fact_by_id(self, fact_id: int) -> dict | None:
        fact_json = await redis_client.hget(self.redis_key_entries, str(fact_id))
        return json.loads(fact_json) if fact_json else None

    async def hard_delete(self, fact_id: int) -> bool:
        fact_id_str = str(fact_id)
        try:
            await asyncio.to_thread(pinecone_index.delete, ids=[fact_id_str], namespace=self.user_id)
            await redis_client.hdel(self.redis_key_entries, fact_id_str)
            await redis_client.lrem(self.redis_key_queue, 0, fact_id_str)
            logger.info(f"Запись ID {fact_id_str} полностью удалена для user {self.user_id}.")
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении записи ID {fact_id_str} для user {self.user_id}: {e}")
            return False
            
    async def recategorize(self, fact_id: int) -> bool:
        fact_id_str = str(fact_id)
        try:
            fact_json = await redis_client.hget(self.redis_key_entries, fact_id_str)
            if not fact_json:
                logger.warning(f"Попытка рекатегоризации несуществующего факта ID {fact_id_str}.")
                return False

            entry_data = json.loads(fact_json)

            if entry_data.get("type") == "contextual_memory":
                return True

            entry_data["type"] = "contextual_memory"
            await redis_client.hset(self.redis_key_entries, fact_id_str, json.dumps(entry_data))
            await redis_client.rpush(self.redis_key_queue, fact_id_str)
            
            logger.info(f"Факт ID {fact_id_str} рекатегоризован и добавлен в очередь на удаление.")
            return True

        except Exception as e:
            logger.error(f"Ошибка при рекатегоризации факта ID {fact_id_str}: {e}")
            return False
