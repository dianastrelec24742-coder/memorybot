
# Файл: memory_core.py (Версия для Render/Cloud)

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

# Загрузка модели по-прежнему происходит один раз при старте
try:
    logger.info("Загрузка модели SentenceTransformer...")
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    VECTOR_DIMENSION = MODEL.get_sentence_embedding_dimension()
    logger.info("Модель SentenceTransformer загружена.")
except Exception as e:
    logger.critical(f"Не удалось загрузить модель SentenceTransformer: {e}")
    # Это критическая ошибка, без модели бот работать не сможет
    exit()

# --- Подключение к облачным сервисам ---
try:
    # Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    
    # Upstash Redis
    UPSTASH_URL = os.getenv("UPSTASH_REDIS_REST_URL")
    UPSTASH_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, UPSTASH_URL, UPSTASH_TOKEN]):
        raise ValueError("Одна или несколько переменных окружения для Pinecone/Upstash не установлены!")

    # Инициализируем Pinecone синхронно при старте
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Проверяем, существует ли индекс, и создаем, если нет
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        logger.warning(f"Индекс Pinecone '{PINECONE_INDEX_NAME}' не найден. Создаю новый...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="cosine", # косинусное расстояние хорошо подходит для семантического поиска
            spec=PodSpec(environment=PINECONE_ENVIRONMENT)
        )
        logger.info(f"Индекс '{PINECONE_INDEX_NAME}' успешно создан.")
    
    # Получаем объект для работы с индексом
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Инициализируем Redis асинхронно
    redis_client = Redis.from_url(UPSTASH_URL, password=UPSTASH_TOKEN, decode_responses=True)

except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА при инициализации облачных сервисов: {e}")
    exit()

class MemoryCore:
    """
    Асинхронная версия для управления LTM с использованием Pinecone и Redis.
    """

    def init(self, user_id, limit=1000):
        self.user_id = str(user_id)
        self.limit = limit
        # Ключи для хранения данных в Redis
        self.redis_key_entries = f"ltm:{self.user_id}:entries" # Hash для хранения всех фактов
        self.redis_key_queue = f"ltm:{self.user_id}:queue"   # List для очереди на удаление
        self.redis_key_id_counter = f"ltm:{self.user_id}:next_id" # Счетчик ID

    @classmethod
    async def create(cls, user_id, limit=1000):
        """Фабричный метод для асинхронного создания и инициализации памяти."""
        instance = cls(user_id, limit)
        await instance._initialize_counter()
        return instance

    async def _initialize_counter(self):
        """Устанавливает счетчик ID в Redis, если он еще не существует."""
        await redis_client.setnx(self.redis_key_id_counter, 0)

    async def _get_next_id(self):
        """Атомарно увеличивает и возвращает новый ID."""
        return await redis_client.incr(self.redis_key_id_counter)

    async def add_to_memory(self, text_chunk: str, is_eternal: bool = False, key: str = None) -> tuple[bool, int]:
        try:
            new_id = await self._get_next_id()
            
            # Логика удаления старых записей, если лимит превышен
            if not is_eternal:
                queue_length = await redis_client.llen(self.redis_key_queue)
                if queue_length >= self.limit:
                    # Извлекаем самый старый ID из очереди
                    oldest_id_str = await redis_client.lpop(self.redis_key_queue)
                    if oldest_id_str:
                        # Удаляем из Pinecone и Redis
                        await pinecone_index.delete(ids=[oldest_id_str], namespace=self.user_id)
                        await redis_client.hdel(self.redis_key_entries, oldest_id_str)
                        logger.info(f"Лимит достигнут. Удалено временное воспоминание с ID {oldest_id_str} для user {self.user_id}")
            
            # Кодирование текста (тяжелая операция, выносим в поток)
            vector = await asyncio.to_thread(MODEL.encode, [text_chunk])

            # Готовим данные для хранения
            entry_data = {
                "id": new_id,
                "timestamp": datetime.now().isoformat(),
                "content": text_chunk,
                "key": key,
                "type": "eternal_fact" if is_eternal else "contextual_memory"
            }
            
            # 1. Сохраняем вектор в Pinecone
            # ID должны быть строками. namespace позволяет изолировать векторы разных пользователей
            await asyncio.to_thread(
                pinecone_index.upsert,vectors=[{"id": str(new_id), "values": vector[0].tolist()}],
                namespace=self.user_id
            )

            # 2. Сохраняем метаданные в Redis
            await redis_client.hset(self.redis_key_entries, str(new_id), json.dumps(entry_data))
            
            # 3. Если запись не вечная, добавляем ее в очередь на удаление
            if not is_eternal:
                await redis_client.rpush(self.redis_key_queue, str(new_id))

            logger.info(f"Успешно добавлена запись ID {new_id} для user {self.user_id} (вечная: {is_eternal}).")
            return True, new_id

        except Exception as e:
            logger.error(f"Не удалось добавить запись в память для user {self.user_id}: {e}", exc_info=True)
            return False, -1

    async def search_in_memory(self, query_text: str, k: int = 5, distance_threshold: float = 0.3, search_type: str = "all") -> list:
        try:
            # 1. Кодируем запрос и ищем в Pinecone
            query_vector = await asyncio.to_thread(MODEL.encode, [query_text])
            
            # Фильтр для поиска только вечных фактов
            pinecone_filter = None
            if search_type == "eternal_only":
                # Pinecone не поддерживает фильтрацию по метаданным в бесплатном тарифе.
                # Поэтому мы делаем "пост-фильтрацию" после получения результатов.
                # Запросим больше кандидатов (k*3), чтобы повысить шанс найти вечные.
                k = k * 3

            search_response = await asyncio.to_thread(
                pinecone_index.query,
                vector=query_vector[0].tolist(),
                top_k=k,
                namespace=self.user_id,
                filter=pinecone_filter
            )

            # 2. Получаем ID найденных векторов
            match_ids = [match['id'] for match in search_response['matches'] if match['score'] >= (1 - distance_threshold)]
            if not match_ids:
                return []

            # 3. Запрашиваем метаданные для этих ID из Redis
            retrieved_entries_json = await redis_client.hmget(self.redis_key_entries, match_ids)
            
            relevant_memories = []
            for entry_json in retrieved_entries_json:
                if entry_json:
                    entry_data = json.loads(entry_json)
                    # Пост-фильтрация для вечных фактов
                    if search_type == "eternal_only" and entry_data.get("type") != "eternal_fact":
                        continue
                    relevant_memories.append(entry_data)

            # Сортировка по актуальности (от новых к старым)
            relevant_memories.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return relevant_memories

        except Exception as e:
            logger.error(f"Ошибка при поиске в памяти для user {self.user_id}: {e}", exc_info=True)
            return []
            
    # Эти методы теперь работают с Redis
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
            # Удаляем из Pinecone
            await asyncio.to_thread(pinecone_index.delete, ids=[fact_id_str], namespace=self.user_id)
            # Удаляем из Redis (из хэша и из очереди)
            await redis_client.hdel(self.redis_key_entries, fact_id_str)
            await redis_client.lrem(self.redis_key_queue, 0, fact_id_str)
            logger.info(f"Запись ID {fact_id_str} полностью удалена для user {self.user_id}.")
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении записи ID {fact_id_str} для user {self.user_id}: {e}")
            return False
            
        # Этот код нужно будет просто добавить в класс MemoryCore в вашем файле

    async def recategorize(self, fact_id: int) -> bool:
        """Меняет тип факта на 'contextual_memory' и добавляет в очередь на удаление."""
        fact_id_str = str(fact_id)
        try:
            # 1. Получаем текущую запись из Redis
            fact_json = await redis_client.hget(self.redis_key_entries, fact_id_str)
            if not fact_json:
                logger.warning(f"Попытка рекатегоризации несуществующего факта ID {fact_id_str}.")
                return False

            entry_data = json.loads(fact_json)

            # 2. Если он уже временный, ничего не делаем
            if entry_data.get("type") == "contextual_memory":
                return True

            # 3. Меняем тип и сохраняем обратно в Redis
            entry_data["type"] = "contextual_memory"
            await redis_client.hset(self.redis_key_entries, fact_id_str, json.dumps(entry_data))

            # 4. Добавляем ID в конец очереди на удаление
            await redis_client.rpush(self.redis_key_queue, fact_id_str)
            
            logger.info(f"Факт ID {fact_id_str} рекатегоризован и добавлен в очередь на удаление.")
            return True

        except Exception as e:
            logger.error(f"Ошибка при рекатегоризации факта ID {fact_id_str}: {e}")
            return False
    


'''
предыдущая версия
# Файл: memory_core.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
from datetime import datetime
import asyncio # <-- Добавили asyncio
import logging

# --- Конфигурация ---
# Эти операции тяжелые, но их нужно сделать лишь один раз при старте.
print("Загрузка модели SentenceTransformer... (это может занять время)")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
VECTOR_DIMENSION = MODEL.get_sentence_embedding_dimension()
print("Модель SentenceTransformer загружена.")
logger = logging.getLogger(__name__)

class MemoryCore:
    """
    Асинхронная версия для управления долгосрочной памятью (LTM).
    Все операции с файлами и тяжелые вычисления выполняются в отдельных потоках,
    не блокируя основной цикл работы бота.
    """

    # Конструктор __init__ НЕ МОЖЕТ быть асинхронным.
    # Поэтому мы делаем его максимально "легким", только инициализируем переменные.
    def __init__(self, user_id, memory_dir="user_memories", limit=1000):
        self.user_id = user_id
        self.limit = limit
        self.index = None
        self.data = None
        self.next_id = 0

        if not os.path.exists(memory_dir):
            os.makedirs(memory_dir)

        self.index_file = os.path.join(memory_dir, f"{user_id}_ltm.index")
        self.data_file = os.path.join(memory_dir, f"{user_id}_ltm.json")

    # Это наш новый "асинхронный конструктор". 
    # Мы будем создавать объект именно через него.
    @classmethod
    async def create(cls, user_id, memory_dir="user_memories", limit=1000):
        """Фабричный метод для асинхронного создания и загрузки памяти."""
        instance = cls(user_id, memory_dir, limit)
        await instance._load_memory()
        return instance

    # Вставь этот код вместо существующего метода _load_memory в НОВОМ файле

    async def _load_memory(self):
        """Асинхронно загружает память из файлов.""" # <-- ВОТ ИСПРАВЛЕННЫЙ ОТСТУП
        if os.path.exists(self.data_file):
            print(f"Загрузка данных для пользователя {self.user_id}...")
            
            def load_json_data():
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            self.data = await asyncio.to_thread(load_json_data)

            # Проверяем, существует ли .index файл. Если да, просто загружаем его.
            if os.path.exists(self.index_file):
                print(f"Загрузка существующего индекса...")
                self.index = await asyncio.to_thread(faiss.read_index, self.index_file)
            
            # Если .index файла НЕТ, то мы его создаем и индексируем данные
            else:
                print("Существующий индекс не найден, создается новый...")
                base_index = faiss.IndexFlatL2(VECTOR_DIMENSION)
                self.index = faiss.IndexIDMap(base_index)
                
                # Проверяем, есть ли что индексировать в .json файле
                if self.data and self.data.get('entries'):
                    print(f"Индексация ВСЕХ {len(self.data['entries'])} записей...")
                    
                    ids_to_index = [int(k) for k in self.data['entries'].keys()]
                    content_to_index = [v['content'] for v in self.data['entries'].values()]

                    if ids_to_index:
                        print(f"Найдено {len(ids_to_index)} ID для индексации.")
                        vectors = await asyncio.to_thread(MODEL.encode, content_to_index)
                        self.index.add_with_ids(vectors.astype('float32'), np.array(ids_to_index, dtype=np.int64))
                
                # ВАЖНО: Сохраняем НОВЫЙ индекс на диск, чтобы не делать этого снова
                print("Сохранение нового индекса...")
                await asyncio.to_thread(faiss.write_index, self.index,self.index_file)

        # Если вообще никаких файлов нет, создаем все с нуля
        else:
            print(f"Создание новой асинхронной памяти для пользователя {self.user_id}...")
            base_index = faiss.IndexFlatL2(VECTOR_DIMENSION)
            self.index = faiss.IndexIDMap(base_index)
            self.data = {'entries': {}, 'id_queue': []}
            # Сразу сохраняем пустые файлы, чтобы они появились
            await self._save_memory() 

        # Вычисляем следующий ID
        all_ids = [int(k) for k in self.data.get('entries', {}).keys()]
        self.next_id = max(all_ids) + 1 if all_ids else 0
        print(f"Память для {self.user_id} готова. Всего записей в индексе: {self.index.ntotal}")



    async def _save_memory(self):
        """Асинхронно сохраняет память в файлы."""

        def save_all():
            faiss.write_index(self.index, self.index_file)
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=4)

        await asyncio.to_thread(save_all)
        print(f"DEBUG: Память для {self.user_id} асинхронно сохранена. Записей: {self.index.ntotal}")

    async def add_to_memory(self, text_chunk: str, is_eternal: bool = False, key: str = None) -> tuple[bool, int]:
        try:
            new_id = self.next_id
            
            # Если это НЕ вечная запись, применяем логику очереди и лимита
            if not is_eternal:
                if len(self.data['id_queue']) >= self.limit:
                    oldest_id = self.data['id_queue'].pop(0)
                    try:
                        self.index.remove_ids(np.array([oldest_id]))
                        del self.data['entries'][str(oldest_id)]
                        logger.info(f"Лимит достигнут. Удалено временное воспоминание с ID {oldest_id}")
                    except Exception as e:
                        logger.error(f"Ошибка при удалении старого ID {oldest_id}: {e}")

            # Кодирование текста
            vector = await asyncio.to_thread(MODEL.encode, [text_chunk])
            vector = vector[0].astype('float32')

            # Добавляем в FAISS и в словарь entries
            self.index.add_with_ids(np.array([vector]), np.array([new_id]))
            self.data['entries'][str(new_id)] = {
                "id": new_id,
                "timestamp": datetime.now().isoformat(),
                "content": text_chunk,
                "key": key,
                "type": "eternal_fact" if is_eternal else "contextual_memory"
            }

            # Если это НЕ вечная запись, добавляем ее ID в очередь
            if not is_eternal:
                self.data['id_queue'].append(new_id)

            self.next_id += 1
            await self._save_memory()
            logger.info(f"Успешно добавлена запись ID {new_id} (вечная: {is_eternal}).")
            return True, new_id
            
        except Exception as e:
            logger.error(f"Не удалось добавить запись в память: {e}", exc_info=True)
            return False, -1
    

    async def search_in_memory(self, query_text: str, k: int = 5, distance_threshold: float = 0.3) -> list:
        """
        Асинхронно ищет k наиболее релевантных воспоминаний...
        """
        if self.index.ntotal == 0:
            return []

        # Кодирование и поиск - тяжелые операции, выносим в поток
        def search_blocking():
            query_vector = MODEL.encode([query_text])[0].astype('float32')
            # Запрашиваем k ближайших соседей
            return self.index.search(np.array([query_vector]), k)

        distances, ids = await asyncio.to_thread(search_blocking)

        relevant_memories= []
        # Итерируемся по результатам, получая и расстояние, и ID
        for dist, i in zip(distances[0], ids[0]):
            # Шаг 1: Фильтрация по порогу релевантности
            if i != -1 and dist <= distance_threshold:
                memory_entry = self.data['entries'][str(i)]
                # Добавляем расстояние к объекту, это может быть полезно для отладки
                memory_entry['distance'] = float(dist) 
                relevant_memories.append(memory_entry)
    
        # Шаг 2: Сортировка по актуальности (от новых к старым)
        relevant_memories.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

        return relevant_memories
        
    async def hard_delete(self, fact_id: int) -> bool:
        """Полностью удаляет запись из памяти, работая с IndexIDMap."""
        if str(fact_id) not in self.data['entries']:
            print(f"DEBUG: Попытка удалить несуществующую запись с ID {fact_id}.")
            return False
        
        try:
            # Шаг 1: Удаляем из индекса FAISS по ID
            self.index.remove_ids(np.array([fact_id], dtype=np.int64))
        
            # Шаг 2: Удаляем из словаря с данными
            del self.data['entries'][str(fact_id)]
        
            # Шаг 3: Если ID был в очереди на удаление, убираем и оттуда
            if fact_id in self.data['id_queue']:
                self.data['id_queue'].remove(fact_id)
        
            await self._save_memory()
            print(f"DEBUG: Запись ID {fact_id} была полностью удалена.")
            return True
        
        except Exception as e:
            print(f"ERROR: Ошибка при полном удалении записи ID {fact_id}: {e}")
            return False
            
    def get_eternal_list(self) -> list:
        """Возвращает список всех вечных записей ('eternal_fact')."""
        eternal_facts = []
        # Итерируемся по словарю entries
        for entry_id, entry_data in self.data['entries'].items():
            if entry_data.get("type") == "eternal_fact":
                eternal_facts.append(entry_data)
        # Сортируем по ID для красивого вывода
        return sorted(eternal_facts, key=lambda x: x['id'])
        
    def get_fact_by_id(self, fact_id: int) -> dict | None:
        """Возвращает данные факта по его ID или None, если не найден."""
        return self.data['entries'].get(str(fact_id))
        
    async def recategorize(self, fact_id: int) -> bool:
        """
        Меняет тип факта с 'eternal_fact' на 'contextual_memory'.
        Факт после этого попадает в общую очередь на удаление.
        """
        fact_id_str = str(fact_id)
        if fact_id_str not in self.data['entries']:
            print(f"DEBUG: Попытка рекатегоризации несуществующего факта ID {fact_id}.")
            return False
    
        entry = self.data['entries'][fact_id_str]
    
        # Если он уже обычный или ID уже в очереди, ничего не делаем
        if entry.get('type') == 'contextual_memory' or fact_id in self.data['id_queue']:
            print(f"DEBUG: Факт {fact_id} уже является обычным.")
            return True # Считаем операцию успешной

        # Меняем тип и добавляем в очередь на удаление
        entry['type'] = 'contextual_memory'
        self.data['id_queue'].append(fact_id)
        print(f"DEBUG: Факт {fact_id} сделан обычным и добавлен в очередь на удаление.")
    
        # Применяем лимит, если он был превышен
        while len(self.data['id_queue']) > self.limit:
            oldest_id = self.data['id_queue'].pop(0)
            try:
                self.index.remove_ids(np.array([oldest_id]))
                del self.data['entries'][str(oldest_id)]
                print(f"DEBUG: Лимит превышен после рекатегоризации. Удалено старое воспоминание с ID {oldest_id}")
            except Exception as e:
                print(f"ERROR: Ошибка при удалении старого ID {oldest_id} после рекатегоризации: {e}")
            
        await self._save_memory()
        return True    
    
    async def search_in_eternal_memory(self, query_text: str, k: int = 5, distance_threshold: float = 0.5) -> list:
        """
        Выполняет прицельный поиск ТОЛЬКО среди вечных фактов ('eternal_fact').
        """
        if self.index.ntotal == 0:
            return []

        # Шаг 1: Выполняем обычный поиск, чтобы получить k ближайших кандидатов из всего индекса
        def search_blocking():
            query_vector = MODEL.encode([query_text])[0].astype('float32')
            # Запрашиваем больше кандидатов (k*3), чтобы повысить шанс найти вечные
            return self.index.search(np.array([query_vector]), k * 3)

        distances, ids = await asyncio.to_thread(search_blocking)

        # Шаг 2: Фильтруем результаты, оставляя только вечные факты
        eternal_memories = []
        for dist, i in zip(distances[0], ids[0]):
            # Проверяем, что ID валиден и прошел порог
            if i != -1 and dist <= distance_threshold:
                memory_entry = self.data['entries'].get(str(i))
                # САМЫЙ ГЛАВНЫЙ ШАГ: Проверяем тип воспоминания
                if memory_entry and memory_entry.get("type") == "eternal_fact":
                    memory_entry['distance'] = float(dist)
                    eternal_memories.append(memory_entry)
        
        # Шаг 3: Сортируем найденные вечные факты по релевантности (от самого близкого к далекому)
        eternal_memories.sort(key=lambda x: x['distance'])

        # Возвращаем только k лучших из найденных
        return eternal_memories[:k]   
 
'''
    


        
        
