# main(cloud).py - Финальная версия для работы с Pinecone/Redis

import logging
import os
import re
import asyncio
import json
import telegram
import subprocess
from io import BytesIO
from telegram.constants import ParseMode
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import speech_recognition as sr
from telegram import Update
from telegram.helpers import escape_markdown
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from collections import defaultdict
from localization import COMMAND_ALIAS_MAP, ETERNAL_SEARCH_TRIGGERS

# Импортируем наш облачный модуль долгосрочной памяти
from memory_core import MemoryCore

# ### ДОБАВЛЕНО ### Импортируем наш новый файл конфигурации
import config

# --- ИНИЦИАЛИЗАЦИЯ И НАСТРОЙКИ ---
user_locks = defaultdict(asyncio.Lock)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- ИНИЦИАЛИЗАЦИЯ API ---
try:
    genai.configure(api_key=config.GOOGLE_API_KEY)
    safety_settings = {
         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
         HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }
    router_model = genai.GenerativeModel(config.ROUTER_MODEL)
    logger.info("Успешная инициализация Google Gemini API.")
    logger.info("Модуль облачной долгосрочной памяти готов к работе.")
except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось настроить Google Gemini API. Проверь ключ. Ошибка: {e}")
    exit()

# --- ФУНКЦИИ-ПОМОЩНИКИ (без изменений) ---

def format_and_escape(text: str) -> str:
    pattern = r'(\\*\\*.*?\\*\\*|__.*?__|```.*?```)'
    result_parts = []
    last_end = 0
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        start_index = match.start()
        unescaped_part = text[last_end:start_index]
        result_parts.append(escape_markdown(unescaped_part, version=2))
        part = match.group(0)
        if part.startswith('**') and part.endswith('**'):
            content = part[2:-2]
            escaped_content = escape_markdown(content, version=2)
            result_parts.append(f'*{escaped_content}*')
        elif part.startswith('__') and part.endswith('__'):
            content = part[2:-2]
            escaped_content = escape_markdown(content, version=2)
            result_parts.append(f'_{escaped_content}_')
        elif part.startswith('```') and part.endswith('```'):
            result_parts.append(part)
        last_end = match.end()
    remaining_part = text[last_end:]
    result_parts.append(escape_markdown(remaining_part, version=2))
    return "".join(result_parts)

async def soften_user_prompt_if_needed(user_prompt: str, user_id: int) -> str:
    try:
        softener_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""
        Проанализируй следующий запрос пользователя на предмет нарушения политики "откровенно сексуального контента".
        ЗАПРОС: "{user_prompt}"
        ПРАВИЛА:
        1. Если запрос НЕ НАРУШАЕТ политику и является приемлемым, верни его в исходном виде, добавив в начало префикс [OK].
        2. Если запрос СЛИШКОМ откровенный и с высокой вероятностью будет заблокирован, перепиши его, сохраняя основное желание и намерение, но используя более мягкие, метафоричные или романтические формулировки. Не добавляй префикс.
        Твой ответ:
        """
        response = await softener_model.generate_content_async(prompt)
        decision = response.text.strip()
        if decision.startswith("[OK]"):
            logger.info(f"Смягчитель: запрос от {user_id} в норме, не требует изменений.")
            return decision[4:]
        else:
            logger.warning(f"Смягчитель: запрос от {user_id} был смягчен. Оригинал: '{user_prompt}'. Новый: '{decision}'")
            return decision
    except Exception as e:
        logger.error(f"Ошибка в функции смягчения запроса: {e}. Возвращаю оригинальный запрос.")
        return user_prompt

async def generate_with_retry(model_function, *args, **kwargs):
    max_retries = 3
    base_delay = 1
    for attempt in range(max_retries):
        try:
            response = await model_function(*args, **kwargs)
            # Добавим проверку: если ответ есть, но он пустой (без 'parts' и 'candidates'), 
            # это тоже может быть проблемой. Считаем это неудачей.
            if response and not response.parts and not response.candidates:
                logger.warning(f"API вернул пустой, но валидный ответ (попытка {attempt + 1}/{max_retries}). Повторяем...")
                if attempt + 1 == max_retries:
                    logger.error("Все попытки исчерпаны. API стабильно возвращает пустой ответ.")
                    return None # Возвращаем None вместо падения
                await asyncio.sleep(base_delay)
                base_delay *= 2
                continue
            return response # Успешный выход
            
        except Exception as e:
            error_str = str(e).lower()
            # Проверяем на ошибки, которые стоит повторить
            if "500" in error_str or "internal error" in error_str or "service unavailable" in error_str:
                logger.warning(f"Ошибка API (попытка {attempt + 1}/{max_retries}): {e}. Повтор через {base_delay} сек.")
                if attempt + 1 == max_retries:
                    logger.error("Все попытки исчерпаны. Не удалось получить ответ от API.")
                    return None # Возвращаем None, а не падаем
                await asyncio.sleep(base_delay)
                base_delay *= 2
            else:
                # Если ошибка другая (например, 400 Bad Request, ValueError), не повторяем, а сразу выходим
                logger.error(f"Неустранимая ошибка API: {e}")
                return None # <<< ГЛАВНОЕ ИЗМЕНЕНИЕ: Возвращаем None, а не делаем 'raise e'
                
        return None
async def summarize_history(history_chunk: list) -> str:
    if not history_chunk: return ""
    logger.info("Запущена функция создания конспекта истории...")
    text_to_summarize = "\
".join([f"{msg['role']}: {' '.join(str(p) for p in msg.get('parts', []))}" for msg in history_chunk])
    try:
        summarizer_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"Ниже приведен фрагмент диалога. Напиши очень краткое содержание этого разговора в 1-2 предложениях, сохранив ключевые факты и намерения.\n\nФрагмент:\n---\n{text_to_summarize}\n---\n\nКраткое содержание:"
        response = await summarizer_model.generate_content_async(prompt)
        summary = response.text.strip()
        logger.info(f"Конспект успешно создан: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Ошибка при создании конспекта: {e}")
        return ""

### УДАЛЕНО ###
# Функции save_history и load_history удалены, так как они несовместимы с облачным хостингом.
# Краткосрочная история теперь хранится только в оперативной памяти (в context.user_data).

async def send_long_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, parse_mode: str = 'MarkdownV2'):
    if not text or not text.strip():
        logger.warning("Попытка отправить пустое сообщение была перехвачена и отменена в send_long_message.")
        return
    MAX_LENGTH = 4096
    parts = []
    text_to_split = text
    while text_to_split:
        if len(text_to_split) <= MAX_LENGTH:
            parts.append(text_to_split)
            break
        chunk = text_to_split[:MAX_LENGTH]
        last_newline = chunk.rfind('\
')
        if last_newline != -1:
            part_to_send = chunk[:last_newline]
            text_to_split = text_to_split[last_newline + 1:]
        else:
            part_to_send = chunk
            text_to_split = text_to_split[MAX_LENGTH:]
        parts.append(part_to_send)
    try:
        for part in parts:
            if not part.strip(): continue
            formatted_part = format_and_escape(part)
            await context.bot.send_message(chat_id=chat_id, text=formatted_part, parse_mode=parse_mode)
            await asyncio.sleep(0.1)
    except telegram.error.BadRequest as e:
        if "can't parse entities" in str(e):
            logger.warning(f"Критическая ошибка парсинга MarkdownV2: {e}. Отправляю как обычный текст.")
            for part in parts:
                if not part.strip(): continue
                await context.bot.send_message(chat_id=chat_id, text=part)
                await asyncio.sleep(0.1)
        else:
            logger.error(f"Неизвестная ошибка BadRequest при отправке сообщения: {e}", exc_info=True)
            await context.bot.send_message(chat_id=chat_id, text="Произошла ошибка при форматировании ответа.")

async def router_check(user_input: str, chat_history: list) -> bool:
    if not user_input: return False
    history_str = "\n".join([f"{msg['role']}: {msg['parts'][0]}" for msg in chat_history[-4:]])
    prompt = f"""Ты — ИИ-роутер. Проанализируй ПОСЛЕДНИЙ ЗАПРОС пользователя в контексте диалога и реши, нужно ли искать информацию в долгосрочной памяти.
Правила:
1. Если запрос — общий вопрос, приветствие, прощание, благодарность, не требующее воспоминаний, ответь: context_sufficient
2. Если пользователь прямо просит что-то вспомнить, спрашивает о прошлых событиях, фактах, идеях, ответь: search_needed.
История диалога:
{history_str}
ПОСЛЕДНИЙ ЗАПРОС: "{user_input}"
Нужен поиск в долгосрочной памяти? Ответь ТОЛЬКО 'search_needed' или 'context_sufficient'.
"""
    try:
        response = await generate_with_retry(router_model.generate_content_async, prompt)
        
        # --- ВОТ ЭТА ПРОВЕРКА ВСЁ РЕШАЕТ ---
        if not response or not response.parts:
            logger.error(f"[ROUTER] Не удалось получить ответ от API. По умолчанию считаем, что поиск нужен.")
            return True # Безопасный выход, если API не ответил

        decision = response.text.strip().lower()
        logger.info(f"[ROUTER] Решение: {decision} для запроса: '{user_input}'")
        return decision == 'search_needed'
    except Exception as e:
        logger.error(f"[ROUTER] Ошибка: {e}. По умолчанию считаем, что поиск нужен.")
        return True

# --- ОБРАБОТЧИКИ КОМАНД УПРАВЛЕНИЯ ПАМЯТЬЮ (адаптированы) ---

async def show_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if 'ltm' not in context.user_data:
        context.user_data['ltm'] = await MemoryCore.create(user_id=user_id, limit=config.LTM_SLOT_LIMIT)
    ltm = context.user_data['ltm']

    ### ИЗМЕНЕНИЕ: Добавлен `await`, так как метод стал асинхронным
    eternal_facts = await ltm.get_eternal_list()
    
    if not eternal_facts:
        await update.message.reply_text("В моей вечной памяти пока нет ни одного факта.")
        return

    response_text = "*Вечные факты в моей памяти:*\
\
"
    for fact in eternal_facts:
        key_str = f"_(ключ: `{fact['key']}`)_" if fact.get('key') else ""
        content_escaped = escape_markdown(fact.get('content', 'Нет содержимого'), version=2)
        response_text += f"*{fact['id']}*: {content_escaped} {key_str}\
"

    await send_long_message(context, update.effective_chat.id, response_text)

async def remember_fact(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not context.args or len(context.args) < 2:
        await update.message.reply_text( "Неверный формат. Используйте: `/запомни <ключ> <текст факта>`")
        return

    key = context.args[0]
    content = " ".join(context.args[1:])
    if 'ltm' not in context.user_data:
        context.user_data['ltm'] = await MemoryCore.create(user_id=user_id, limit=config.LTM_SLOT_LIMIT)
    ltm = context.user_data['ltm']

    success, new_id = await ltm.add_to_memory(text_chunk=content, key=key, is_eternal=True)
    if success:
        await update.message.reply_text(f"Хорошо, я навсегда запомнил. Факт с ID `{new_id}` и ключом `{key}` сохранен.")
    else:
        await update.message.reply_text(f"Не удалось сохранить факт.")

async def forget_fact(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Неверный формат. Используйте: `/забудь <ID факта>`")
        return

    fact_id = int(context.args[0])
    if 'ltm' not in context.user_data:
        context.user_data['ltm'] = await MemoryCore.create(user_id=user_id, limit=config.LTM_SLOT_LIMIT)
    ltm = context.user_data['ltm']
    
    ### ИЗМЕНЕНИЕ: Добавлен `await`, так как метод стал асинхронным
    fact_to_delete = await ltm.get_fact_by_id(fact_id)
    if not fact_to_delete:
        await update.message.reply_text(f"Факт с ID `{fact_id}` не найден.")
        return

    success = await ltm.hard_delete(fact_id)
    if success:
        content_escaped = escape_markdown(fact_to_delete.get('content', ''), version=2)
        await update.message.reply_text(f"Факт `{fact_id}` ('{content_escaped}') был навсегда удален.")
    else:
        await update.message.reply_text(f"Не удалось удалить факт с ID `{fact_id}`.")

async def recategorize_fact(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Неверный формат. Используйте: `/сделай_обычным <ID факта>`")
        return
        
    fact_id = int(context.args[0])
    if 'ltm' not in context.user_data:
        context.user_data['ltm'] = await MemoryCore.create(user_id=user_id, limit=config.LTM_SLOT_LIMIT)
    ltm = context.user_data['ltm']

    ### ИЗМЕНЕНИЕ: Добавлен `await`, так как метод стал асинхронным
    fact = await ltm.get_fact_by_id(fact_id)
    if not fact:
        await update.message.reply_text(f"Факт с ID `{fact_id}` не найден.")
        return
    
    if fact.get('type') != 'eternal_fact':
        await update.message.reply_text(f"Факт `{fact_id}` уже является обычным.")
        return
        
    success = await ltm.recategorize(fact_id)
    if success:
        content_escaped = escape_markdown(fact.get('content', ''), version=2)
        await update.message.reply_text(f"Хорошо, факт `{fact_id}` ('{content_escaped}') теперь считается обычным знанием и может быть забыт со временем.")
    else:
        await update.message.reply_text(f"Не удалось изменить статус факта `{fact_id}`.")

# --- ОСНОВНАЯ ЛОГИКА ОБРАБОТКИ ЗАПРОСА ---

async def process_user_request(context: ContextTypes.DEFAULT_TYPE, user_id: int, chat_id: int, user_parts: list, original_user_input: str):
    lock = user_locks.setdefault(user_id, asyncio.Lock())
    async with lock:
        try:
            # === ИНИЦИАЛИЗАЦИЯ И ВЫБОР МОДЕЛИ ===
            model_id = config.DEFAULT_GEMINI_MODEL # Пока используем модель по умолчанию
            user_specific_model = genai.GenerativeModel(model_name=model_id, safety_settings=safety_settings)

            if 'ltm' not in context.user_data:
                logger.info(f"Для пользователя {user_id} нет активного объекта ltm. Создаю новый...")
                context.user_data['ltm'] = await MemoryCore.create(user_id=user_id, limit=config.LTM_SLOT_LIMIT)
            ltm = context.user_data['ltm']

            ### ИЗМЕНЕНИЕ: Убран вызов load_history. История просто инициализируется, если ее нет.
            context.user_data.setdefault('history', [])

            # === ПОИСК В ДОЛГОСРОЧНОЙ ПАМЯТИ (LTM) ===
            long_term_context = ""
            found_facts = []
            is_forced_search = any(phrase in original_user_input.lower() for phrase in ETERNAL_SEARCH_TRIGGERS)

            if is_forced_search:
                logger.info(f"[FORCE SEARCH] Активирован принудительный поиск по вечной памяти для {user_id}.")
                cleaned_input = original_user_input
                for phrase in ETERNAL_SEARCH_TRIGGERS: cleaned_input = cleaned_input.lower().replace(phrase, "").strip()
                if not cleaned_input: cleaned_input = original_user_input
                
                ### ИЗМЕНЕНИЕ: Метод `search_in_eternal_memory` заменен на `search_in_memory` с параметром
                found_facts_raw = await ltm.search_in_memory(cleaned_input, k=3, distance_threshold=0.5, search_type="eternal_only")
                if found_facts_raw: found_facts = [fact['content'] for fact in found_facts_raw]

            elif original_user_input and await router_check(original_user_input, context.user_data['history']):
                logger.info(f"[ROUTER] Принято решение выполнить поиск в LTM для {user_id}.")
                found_facts_raw = await ltm.search_in_memory(original_user_input, k=3, distance_threshold=0.3)
                if found_facts_raw: found_facts = [fact['content'] for fact in found_facts_raw]
            
            if found_facts:
                long_term_context = ". ".join(found_facts)
                logger.info(f"Найдено в LTM для {user_id}: '{long_term_context}'")

            # === ФОРМИРОВАНИЕ ПРОМПТА И ГЕНЕРАЦИЯ ОТВЕТА ===            # --- Шаг 1: Формируем историю для API вызова ОДИН РАЗ ---
            api_call_history = list(context.user_data['history'])
            if long_term_context:
                context_message = f"ОБЯЗАТЕЛЬНЫЙ КОНТЕКСТ: Вот некоторые факты из нашей долгосрочной памяти, которые могут быть релевантны: «{long_term_context}». Используй их для ответа."
                api_call_history.insert(0, {'role': 'model', 'parts': [context_message]})
            
            api_call_history.append({'role': 'user', 'parts': user_parts})
            final_user_input_for_ltm = original_user_input

            # --- Шаг 2: Цикл генерации с корректной логикой ---
            max_attempts = 3
            response = None
            
            for attempt in range(max_attempts):
                logger.info(f"Попытка генерации/коррекции №{attempt + 1}...")
                response = await generate_with_retry(user_specific_model.generate_content_async, api_call_history)

                # --- Единая и четкая логика ---
                # 1. Полный провал API (функция-помощник вернула None)
                if not response:
                    logger.error(f"generate_with_retry вернула None на попытке {attempt + 1}. API не отвечает или вернул ошибку.")
                    # Если это последняя попытка, цикл просто завершится и response останется None
                    if attempt + 1 < max_attempts:
                        await asyncio.sleep(1) # небольшая пауза перед следующей попыткой
                        continue
                    else:
                        break # выходим из цикла с response = None

                # 2. Успешный ответ
                if response.parts:
                    logger.info("Ответ от Gemini успешно получен.")
                    break 

                # 3. Блокировка цензурой (разные типы)
                # Тип 1: Жесткая блокировка (нет даже кандидатов)
                if not response.candidates:
                    logger.warning(f"Цензура [Тип 1: Жесткая блокировка] для {user_id}. Запускаю смягчитель запроса.")
                    last_user_prompt = " ".join(part for part in api_call_history[-1]['parts'] if isinstance(part, str))
                    if last_user_prompt:
                        softened_prompt = await soften_user_prompt_if_needed(last_user_prompt, user_id)
                        api_call_history[-1]['parts'] = [softened_prompt] 
                        final_user_input_for_ltm = softened_prompt
                        continue # Переходим к следующей попытке с новым промптом
                    else:
                        logger.error("Не удалось извлечь текст запроса для смягчения.")
                        response = None # Сбрасываем response, чтобы выйти с ошибкой
                        break
                
                # Тип 2: Блокировка ответа по причине SAFETY
                is_safety_blocked = (response.candidates[0].finish_reason == 3)
                if is_safety_blocked:
                    logger.warning(f"Цензура [Тип 2: Блокировка ответа - SAFETY] для {user_id}. Запускаю самокоррекцию ответа.")
                    correction_prompt = {'role': 'model', 'parts': ["СИСТЕМНАЯ ЗАМЕТКА: Твой предыдущий ответ был заблокирован. Переформулируй."]}
                    api_call_history.append(correction_prompt)
                    continue # Переходим к следующей попытке с просьбой о коррекции

                # 4. Другие причины отсутствия ответа (например, finish_reason = OTHER)
                logger.warning(f"Не удалось получить ответ, но это не блокировка цензуры. Finish reason: {response.candidates[0].finish_reason}")
                response = None # Считаем это провалом
                break

            
            # --- Шаг 3: Обрабатываем результат ПОСЛЕ цикла ---
            bot_response_text_raw = ""
            response_was_successful = False

            if response and response.parts:
                bot_response_text_raw = response.text
                response_was_successful = True
                
                # Обновление истории: добавляем и запрос, и ответ в основную историю
                context.user_data['history'].append({'role': 'user', 'parts': user_parts})
                context.user_data['history'].append({'role': 'model', 'parts': [bot_response_text_raw]})
            else:
                bot_response_text_raw = "Мне почему-то не удалось сформулировать ответ. Возможно, мой внутренний фильтр сработал слишком сильно. Попробуй перефразировать свой запрос."
                logger.error(f"Не удалось сгенерировать ответ для {user_id} после {max_attempts} попыток.")

            # --- Шаг 4: Отправка и сохранение ---
            await send_long_message(context, chat_id, bot_response_text_raw, parse_mode='MarkdownV2')

            if response_was_successful:
                logger.info(f"Сохраняем диалог в долгосрочную память для {user_id}...")
                full_context_chunk = f"Пользователь: {final_user_input_for_ltm}\nGemini: {bot_response_text_raw}"
                await ltm.add_to_memory(full_context_chunk)
                logger.info(f"Сохранение в LTM для {user_id} завершено.")
            

            # === СЖАТИЕ ИСТОРИИ (этот блок выполняется всегда в конце) ===
            if len(context.user_data['history']) > config.MAX_HISTORY_MESSAGES:
                # --> потому что он выполняется только когда if истинно.
                logger.info(f"История для {user_id} слишком длинная, запускаем сжатие...")
                messages_to_keep = config.MAX_HISTORY_MESSAGES - 2
                messages_to_summarize = context.user_data['history'][:-messages_to_keep]
                remaining_history = context.user_data['history'][-messages_to_keep:]
                
                summary_text = await summarize_history(messages_to_summarize)
                new_history = []
                if summary_text:
                    # Этот if тоже имеет дополнительный отступ, так как он вложен
                    new_history.append({'role': 'model', 'parts': [f"Это краткое содержание нашего предыдущего разговора: {summary_text}"]})
                new_history.extend(remaining_history)
                context.user_data['history'] = new_history

        except Exception as e:
            logger.error(f"Произошла критическая ошибка в process_user_request для {user_id}: {e}", exc_info=True)
            await context.bot.send_message(chat_id=chat_id, text="Ой, что-то пошло не так при обработке вашего запроса.")
            
# --- ВХОДНЫЕ ОБРАБОТЧИКИ TELEGRAM (без изменений) ---

async def route_natural_command(update: Update, context: ContextTypes.DEFAULT_TYPE, text_to_check: str = None) -> bool:
    text_from_source = text_to_check if text_to_check is not None else (update.message.text if update.message else None)
    if not text_from_source: return False
    text_lower = text_from_source.lower().strip()
    
    # ### ДОБАВЛЕНО ### Ссылка на глобальный словарь для корректной работы
    global COMMAND_ALIASES
    for alias, handler_func in COMMAND_ALIASES.items():
        if text_lower.startswith(alias + ' ') or text_lower == alias:
            args_str = text_from_source.strip()[len(alias):].strip()
            context.args = args_str.split() if args_str else []
            logger.info(f"Найдена естественная команда: '{alias}', аргументы: {context.args}")
            await handler_func(update, context)
            return True
    return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_html(f"Привет, {update.effective_user.mention_html()}! Я готов к работе.")

async def handle_text_and_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await route_natural_command(update, context): return
    message = update.message
    user_id = update.effective_user.id
    user_text = message.text or message.caption or ""
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    user_parts = [user_text] if user_text else []
    if message.photo:
        photo_file = await message.photo[-1].get_file()
        image_bytes = bytes(await photo_file.download_as_bytearray())
        user_parts.append({"mime_type": "image/jpeg", "data": image_bytes})
    if not user_parts: return
    await process_user_request(context, user_id, message.chat_id, user_parts, user_text)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    user_id = update.effective_user.id
    status_message = await message.reply_text("Обрабатываю ваше голосовое сообщение...")
    oga_filename, wav_filename = f"{message.voice.file_id}.oga", f"{message.voice.file_id}.wav"
    try:
        voice_file = await message.voice.get_file()
        await voice_file.download_to_drive(oga_filename)
        subprocess.run(['ffmpeg', '-i', oga_filename, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav_filename], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        r = sr.Recognizer()
        with sr.AudioFile(wav_filename) as source:
            audio = r.record(source)
            recognized_text = await asyncio.to_thread(r.recognize_google, audio, language='ru-RU')
        logger.info(f"Распознан текст: '{recognized_text}'")
        await status_message.edit_text("Распознал. Думаю...")
        if await route_natural_command(update, context, text_to_check=recognized_text):
            await status_message.delete()
            return
        await status_message.delete()
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        await process_user_request(context, user_id, message.chat_id, [recognized_text], recognized_text)
    except sr.UnknownValueError:
        await status_message.edit_text("К сожалению, я не смог разобрать, что вы сказали.")
    except Exception as e:
        logger.error(f"Ошибка обработки голоса: {e}", exc_info=True)
        await status_message.edit_text("Произошла ошибка при обработке голоса.")
    finally:
        if os.path.exists(oga_filename): os.remove(oga_filename)
        if os.path.exists(wav_filename): os.remove(wav_filename)
        
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    status_message = await message.reply_text("Получаю ваш файл...")
    try:
        document = message.document
        if document.file_size > config.MAX_FILE_SIZE_BYTE:
            await status_message.edit_text(f"⚠️ Файл слишком большой.")
        elif not document.file_name.endswith(('.txt', '.py', '.json', '.md', '.csv', '.html', '.css', '.js', '.xml')):
            await status_message.edit_text(f"⚠️ Извините, я не могу обработать такой тип файла.")
        else:
            await status_message.edit_text("Анализирую текст вашего файла...")
            doc_file = await document.get_file()
            file_bytes = await doc_file.download_as_bytearray()
            file_content = file_bytes.decode('utf-8')
            prompt = f"Проанализируй содержимое файла «{document.file_name}» и дай свой ответ.\n\n--- НАЧАЛО ---\n{file_content}\n--- КОНЕЦ ---"
            await process_user_request(context, message.effective_user.id, message.chat_id, [prompt], prompt)
            await status_message.delete()
    except Exception as e:
        logger.error(f"Не удалось обработать файл: {e}", exc_info=True)
        await status_message.edit_text("Произошла ошибка при обработке вашего файла.")

# --- ЗАПУСК БОТА ---
# ### ДОБАВЛЕНО ### Объявляем словарь здесь, чтобы он был доступен в route_natural_command
COMMAND_ALIASES = {}

def main() -> None:
    application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    handler_map = {
        "CMD_REMEMBER": remember_fact, "CMD_FORGET": forget_fact,
        "CMD_SHOW_MEMORY": show_memory, "CMD_RECATEGORIZE": recategorize_fact,
    }
    
    global COMMAND_ALIASES
    for alias, internal_key in COMMAND_ALIAS_MAP.items():
        if internal_key in handler_map:
            COMMAND_ALIASES[alias] = handler_map[internal_key]

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler(["memory", "show_eternal"], show_memory))
    application.add_handler(CommandHandler("remember", remember_fact))
    application.add_handler(CommandHandler("forget", forget_fact))
    application.add_handler(CommandHandler("make_regular", recategorize_fact))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_and_photo))
    application.add_handler(MessageHandler(filters.PHOTO, handle_text_and_photo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("Бот запускается...")
    application.run_polling()

if __name__ == '__main__':
    main()
    
