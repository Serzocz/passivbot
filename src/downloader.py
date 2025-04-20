import argparse
import asyncio
import datetime
import gzip
import json
import logging
import inspect
import os
import shutil
import sys
import traceback
import zipfile
from collections import deque, defaultdict
from functools import wraps
from io import BytesIO
from pathlib import Path
from time import time
from typing import List, Dict, Any, Tuple
from uuid import uuid4
from urllib.request import urlopen

import aiohttp
import pprint
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import tqdm

from pure_funcs import (
    date_to_ts,
    ts_to_date_utc,
    safe_filename,
    symbol_to_coin,
    get_template_live_config,
    coin_to_symbol,
)
from procedures import (
    make_get_filepath,
    format_end_date,
    utc_ms,
    get_file_mod_utc,
    get_first_timestamps_unified,
    add_arguments_recursively,
    load_config,
)

# ========================= CONFIGURABLES & GLOBALS =========================

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# --- МАКСИМАЛЬНАЯ ЧАСТОТА ЗАПРОСОВ (общее ограничение) ---
MAX_REQUESTS_PER_MINUTE = 120
REQUEST_TIMESTAMPS = deque(maxlen=5000)  # для проверки ограничения по кол-ву запросов

# --- ОГРАНИЧЕНИЕ ПАРАЛЛЕЛЬНЫХ ЗАГРУЗОК ---
MAX_CONCURRENT_DOWNLOADS = 5  # сколько одновременно архивов/дней можно качать
# Можно менять выше/ниже по ситуации

# --- ПАРАМЕТРЫ ПОВТОРНЫХ ПОПЫТОК ---
MAX_RETRIES = 5
BASE_DELAY = 1.0  # сек. базовая задержка перед повтором


# ========================= HELPER FUNCTIONS =========================


def is_valid_date(date):
    try:
        _ = date_to_ts(date)
        return True
    except Exception:
        return False


def get_function_name():
    return inspect.currentframe().f_back.f_code.co_name


def deduplicate_rows(arr: np.ndarray) -> np.ndarray:
    """
    Remove duplicate rows from a 2D NumPy array while preserving order.
    """
    rows_as_tuples = map(tuple, arr)
    seen = set()
    unique_indices = []
    for i, row_tuple in enumerate(rows_as_tuples):
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_indices.append(i)
    return arr[unique_indices]


def dump_ohlcv_data(data, filepath):
    """
    Сериализует данные OHLCV в npy-файл, предварительно убирает дубликаты строк.
    """
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if isinstance(data, pd.DataFrame):
        data = ensure_millis(data[columns]).astype(float).values
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise Exception(f"Unknown data format for {filepath}")
    data = deduplicate_rows(data)
    np.save(filepath, data)


def load_ohlcv_data(filepath: str) -> pd.DataFrame:
    """
    Загрузить npy-файл, убрать дубликаты и вернуть DataFrame c колонками
    ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    arr = np.load(filepath, allow_pickle=True)
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    arr_deduplicated = deduplicate_rows(arr)
    if len(arr) != len(arr_deduplicated):
        dump_ohlcv_data(arr_deduplicated, filepath)
        print(
            f"Caught .npy file with duplicate rows: {filepath} Overwrote with deduplicated version."
        )
    return ensure_millis(pd.DataFrame(arr_deduplicated, columns=columns))


def get_days_in_between(start_day, end_day):
    """
    Сгенерировать список всех дат 'YYYY-MM-DD' от start_day до end_day включительно.
    """
    date_format = "%Y-%m-%d"
    start_date = datetime.datetime.strptime(format_end_date(start_day), date_format)
    end_date = datetime.datetime.strptime(format_end_date(end_day), date_format)
    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date.strftime(date_format))
        current_date += datetime.timedelta(days=1)
    return days


def fill_gaps_in_ohlcvs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполнение пропусков в OHLCV, если интервалы менее критического размера.
    """
    interval = 60000
    new_timestamps = np.arange(df["timestamp"].iloc[0], df["timestamp"].iloc[-1] + interval, interval)
    new_df = df.set_index("timestamp").reindex(new_timestamps)
    new_df.close = new_df.close.ffill()
    for col in ["open", "high", "low"]:
        new_df[col] = new_df[col].fillna(new_df.close)
    new_df["volume"] = new_df["volume"].fillna(0.0)
    return new_df.reset_index().rename(columns={"index": "timestamp"})


def ensure_millis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит колонку 'timestamp' к миллисекундам (если секунды или микросекунды).
    """
    if "timestamp" not in df.columns or df.empty:
        return df
    first_val = df.timestamp.iloc[0]
    if first_val > 1e14:  # microseconds
        df.timestamp /= 1000
    elif first_val > 1e11:  # уже миллисекунды
        pass
    else:  # секунды
        df.timestamp *= 1000
    return df


# ========================= КЛАСС OHLCVManager =========================

class OHLCVManager:
    """
    Класс для управления скачиванием и кешированием OHLCV на разных биржах.
    """

    def __init__(
        self,
        exchange,
        start_date=None,
        end_date=None,
        cc=None,
        gap_tolerance_ohlcvs_minutes=120.0,
        verbose=True,
    ):
        # Подменяем 'binance' на 'binanceusdm' внутри
        self.exchange = "binanceusdm" if exchange == "binance" else exchange
        self.quote = "USDC" if exchange == "hyperliquid" else "USDT"

        self.start_date = "2020-01-01" if start_date is None else format_end_date(start_date)
        self.end_date = format_end_date("now" if end_date is None else end_date)
        self.start_ts = date_to_ts(self.start_date)
        self.end_ts = date_to_ts(self.end_date)

        self.cc = cc  # клиент ccxt.async_support
        self.verbose = verbose
        self.markets = None

        # Настройки кеша
        self.cache_filepaths = {
            "markets": os.path.join("caches", self.exchange, "markets.json"),
            "ohlcvs": os.path.join("historical_data", f"ohlcvs_{self.exchange}"),
            "first_timestamps": os.path.join("caches", self.exchange, "first_timestamps.json"),
        }

        # Лимиты (если нужны разные)
        self.max_requests_per_minute = {"": 30, "gateio": 30}
        self.request_timestamps = deque(maxlen=1000)  # для локального rate-limiting
        self.gap_tolerance_ohlcvs_minutes = gap_tolerance_ohlcvs_minutes

        # Семафор для ограничения параллельных задач (загрузки zips, запросы и т.д.)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    def update_date_range(self, new_start_date=None, new_end_date=None):
        """
        Позволяет динамически менять интервал [start_date, end_date].
        """
        if new_start_date:
            if isinstance(new_start_date, (float, int)):
                self.start_date = ts_to_date_utc(new_start_date)
            elif isinstance(new_start_date, str):
                self.start_date = new_start_date
            else:
                raise Exception(f"invalid start date {new_start_date}")
            self.start_ts = date_to_ts(self.start_date)
        if new_end_date:
            if isinstance(new_end_date, (float, int)):
                self.end_date = ts_to_date_utc(new_end_date)
            elif isinstance(new_end_date, str):
                self.end_date = new_end_date
            else:
                raise Exception(f"invalid end date {new_end_date}")
            self.end_date = format_end_date(self.end_date)
            self.end_ts = date_to_ts(self.end_date)

    def get_symbol(self, coin: str):
        """
        Преобразовать coin 'BTC' в биржевой символ 'BTC/USDT:USDT' (или иной формат)
        """
        assert self.markets, "нужно вызвать load_markets() перед этим"
        return coin_to_symbol(
            coin,
            {k for k in self.markets if self.markets[k].get("swap") and k.endswith(f":{self.quote}")},
            self.quote,
        )

    def has_coin(self, coin: str) -> bool:
        """
        Проверка, торгуется ли coin на данной бирже (учитывая self.quote)
        """
        sym = None
        try:
            sym = self.get_symbol(coin)
        except Exception:
            return False
        return bool(sym)

    def get_market_specific_settings(self, coin: str) -> dict:
        """
        Вытаскиваем настройки контракта (maker_fee, taker_fee, contractSize, precision, limits и т.д.)
        """
        mss = self.markets[self.get_symbol(coin)]
        mss["hedge_mode"] = True
        mss["maker_fee"] = mss["maker"]
        mss["taker_fee"] = mss["taker"]
        mss["c_mult"] = mss["contractSize"]
        mss["min_cost"] = mc if (mc := mss["limits"]["cost"]["min"]) is not None else 0.01
        mss["price_step"] = mss["precision"]["price"]
        mss["min_qty"] = max(
            lm if (lm := mss["limits"]["amount"]["min"]) is not None else 0.0,
            pm if (pm := mss["precision"]["amount"]) is not None else 0.0,
        )
        mss["qty_step"] = mss["precision"]["amount"]

        # Уточняем комиссии для некоторых бирж вручную:
        if self.exchange == "binanceusdm":
            pass
        elif self.exchange == "bybit":
            # ccxt часто даёт неверные комиссии для bybit
            mss["maker"] = mss["maker_fee"] = 0.0002
            mss["taker"] = mss["taker_fee"] = 0.00055
        elif self.exchange == "bitget":
            pass
        elif self.exchange == "gateio":
            # Для gateio perps часто даёт неверные
            mss["maker"] = mss["maker_fee"] = 0.0002
            mss["taker"] = mss["taker_fee"] = 0.0005

        return mss

    def filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Отсеять строки вне интервала [start_ts, end_ts].
        """
        if df.empty:
            return df
        return df[(df.timestamp >= self.start_ts) & (df.timestamp <= self.end_ts)].reset_index(
            drop=True
        )

    async def check_rate_limit(self):
        """
        Простейшая реализация rate-limit для 60сек интервала. Если мы уже сделали
        max_requests_per_minute[self.exchange] запросов — ждём.
        """
        current_time = time()
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()

        mrpm = self.max_requests_per_minute.get(self.exchange, self.max_requests_per_minute[""])
        if len(self.request_timestamps) >= mrpm:
            sleep_time = 60 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                if self.verbose:
                    logging.info(
                        f"{self.exchange} Rate limit reached, sleeping for {sleep_time:.2f} seconds"
                    )
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(time())

    async def _robust_fetch_url(
        self, session: aiohttp.ClientSession, url: str,
        max_retries=MAX_RETRIES, base_delay=BASE_DELAY
    ) -> bytes:
        """
        Выполняет запрос GET к url с:
            - семафором (огранич. параллелизма)
            - rate-limit
            - повторными попытками (retry) при ошибках
        Возвращает bytes контент.
        """
        attempt = 1
        while True:
            await self.check_rate_limit()
            async with self.semaphore:
                try:
                    async with session.get(url) as response:
                        response.raise_for_status()
                        return await response.read()
                except Exception as e:
                    if attempt >= max_retries:
                        logging.error(f"[{self.exchange}] Failed after {attempt} retries: {url}")
                        raise
                    delay = base_delay * 2 ** (attempt - 1)
                    if self.verbose:
                        logging.warning(
                            f"[{self.exchange}] Error fetching {url}: {e}. "
                            f"Retrying {attempt}/{max_retries} in {delay:.1f}s..."
                        )
                    await asyncio.sleep(delay)
                    attempt += 1

    def load_cc(self):
        """
        Инициализировать ccxt-клиент, если ещё не создан.
        """
        if self.cc is None:
            self.cc = getattr(ccxt, self.exchange)({"enableRateLimit": True})
            self.cc.options["defaultType"] = "swap"

    async def load_markets(self):
        """
        Загрузка и кеширование маркетов (markets) для данной биржи.
        """
        self.load_cc()
        self.markets = self.load_markets_from_cache()
        if self.markets:
            return
        self.markets = await self.cc.load_markets()
        self.dump_markets_to_cache()

    def load_markets_from_cache(self, max_age_ms=1000 * 60 * 60 * 24) -> dict:
        """
        Пытаемся загрузить markets из локального json, если его возраст < max_age_ms.
        """
        try:
            fname = self.cache_filepaths["markets"]
            if os.path.exists(fname):
                if utc_ms() - get_file_mod_utc(fname) < max_age_ms:
                    markets = json.load(open(fname))
                    if self.verbose:
                        logging.info(f"{self.exchange} Loaded markets from cache")
                    return markets
            return {}
        except Exception as e:
            logging.error(f"Error with {get_function_name()} {e}")
            return {}

    def dump_markets_to_cache(self):
        """
        Сохранить markets в локальный json-файл.
        """
        if self.markets:
            try:
                filepath = make_get_filepath(self.cache_filepaths["markets"])
                json.dump(self.markets, open(filepath, "w"))
                if self.verbose:
                    logging.info(f"{self.exchange} Dumped markets to cache")
            except Exception as e:
                logging.error(f"Error with {get_function_name()} {e}")

    # ======================================
    # Основные методы для OHLCV (загрузка, кеш, пропуски и т.д.)
    # ======================================

    async def get_ohlcvs(self, coin, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Высокоуровневый метод:
          1) Читает OHLCV из локального кеша (если есть все дни)
          2) Если чего-то не хватает, скачивает
          3) Возвращает итоговый df (при необходимости заполняет мелкие пропуски)
        """
        if not self.markets:
            await self.load_markets()
        if not self.has_coin(coin):
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        if start_date or end_date:
            self.update_date_range(new_start_date=start_date, new_end_date=end_date)

        missing_days = await self.get_missing_days_ohlcvs(coin)
        if missing_days:
            await self.download_ohlcvs(coin)

        ohlcvs = await self.load_ohlcvs_from_cache(coin)
        if not ohlcvs.empty:
            # Преобразуем volume в quote-volume (volume * close)
            ohlcvs["volume"] = ohlcvs["volume"] * ohlcvs["close"]
        return ohlcvs

    async def get_missing_days_ohlcvs(self, coin) -> List[str]:
        """
        Сравнивает список дней, которые нам нужны, со списком npy-файлов на диске.
        Возвращает те дни, которых нет (значит — нужно догружать).
        """
        start_date = await self.get_start_date_modified(coin)
        days = get_days_in_between(start_date, self.end_date)
        dirpath = os.path.join(self.cache_filepaths["ohlcvs"], coin)
        if not os.path.exists(dirpath):
            return days
        all_files = os.listdir(dirpath)
        missing = [d for d in days if (d + ".npy") not in all_files]
        return sorted(missing)

    async def get_start_date_modified(self, coin):
        """
        Учитывает первый таймстемп для конкретной монеты, чтобы не лезть дальше истории.
        """
        fts = await self.get_first_timestamp(coin)
        return ts_to_date_utc(max(self.start_ts, fts))[:10]

    async def download_ohlcvs(self, coin):
        """
        Вызов нужного метода скачивания (binance, bybit, bitget, gateio ...)
        """
        if not self.has_coin(coin):
            return
        if self.exchange == "binanceusdm":
            await self.download_ohlcvs_binance(coin)
        elif self.exchange == "bybit":
            await self.download_ohlcvs_bybit(coin)
        elif self.exchange == "bitget":
            await self.download_ohlcvs_bitget(coin)
        elif self.exchange == "gateio":
            if self.cc is None:
                self.load_cc()
            await self.download_ohlcvs_gateio(coin)
        # и т.д. при необходимости для других бирж

    async def load_ohlcvs_from_cache(self, coin) -> pd.DataFrame:
        """
        Загружает все npy-файлы (месячные + дневные) из кеша и строго проверяет,
        что интервалы по 1 минуте без больших пропусков.
        Если пропуски > gap_tolerance_ohlcvs_minutes — вернёт пустой DF.
        """
        dirpath = os.path.join(self.cache_filepaths["ohlcvs"], coin, "")
        if not os.path.exists(dirpath):
            return pd.DataFrame()

        all_files = sorted([f for f in os.listdir(dirpath) if f.endswith(".npy")])
        all_days = get_days_in_between(self.start_date, self.end_date)
        all_months = sorted(set([x[:7] for x in all_days]))

        # Сперва грузим месячные файлы
        files_to_load = [x for x in all_files if x.replace(".npy", "") in all_months]
        # Затем добавляем дневные, которые не попали в вышее
        files_to_load += [
            x for x in all_files if x.replace(".npy", "") in all_days and x not in files_to_load
        ]

        dfs = []
        for f in files_to_load:
            filepath = os.path.join(dirpath, f)
            try:
                df_part = load_ohlcv_data(filepath)
                dfs.append(df_part)
            except Exception as e:
                logging.error(f"Error loading file {f}: {e}")

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df = self.filter_date_range(df)

        # Проверка пропусков: если где-то разрыв > 1 мин, считаем это «большой пропуск»
        intervals = np.diff(df["timestamp"].values)
        if (intervals != 60000).any():
            greatest_gap = int(intervals.max() / 60000.0)
            if greatest_gap > self.gap_tolerance_ohlcvs_minutes:
                logging.warning(
                    f"[{self.exchange}] Gaps detected in {coin} OHLCV data. "
                    f"Greatest gap: {greatest_gap} minutes. Returning empty DataFrame."
                )
                return pd.DataFrame(columns=df.columns)
            else:
                df = fill_gaps_in_ohlcvs(df)

        return df

    # ============================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ СКАЧИВАНИЯ ИЗ АРХИВОВ
    # ============================================

    def copy_ohlcvs_from_old_dir(self, new_dirpath, old_dirpath, missing_days, coin):
        """
        Если раньше у вас менялась структура папок, этот метод копирует старые npy-файлы.
        """
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        files_copied = 0
        if os.path.exists(old_dirpath):
            for d0 in os.listdir(old_dirpath):
                if d0.endswith(".npy") and d0[:10] in missing_days:
                    src = os.path.join(old_dirpath, d0)
                    dst = os.path.join(new_dirpath, d0)
                    if os.path.exists(dst):
                        continue
                    try:
                        shutil.copy(src, dst)
                        files_copied += 1
                    except Exception as e:
                        logging.error(f"{self.exchange} error copying {src} -> {dst} {e}")
        if files_copied:
            logging.info(f"{self.exchange} copied {files_copied} files from {old_dirpath} to {new_dirpath}")
            return True
        else:
            return False

    async def get_first_timestamp(self, coin: str) -> float:
        """
        Возвращает таймстемп первой доступной свечи (либо из кеша, либо делаем один запрос).
        """
        if (fts := self.load_first_timestamp(coin)) not in [None, 0.0]:
            return fts

        if not self.markets:
            self.load_cc()
            await self.load_markets()

        if not self.has_coin(coin):
            self.dump_first_timestamp(coin, 0.0)
            return 0.0

        # Разные биржи могут иметь логику, как найти 1-й таймстемп
        if self.exchange == "binanceusdm":
            # Возьмём самую раннюю дневную свечу
            ohlcvs = await self.cc.fetch_ohlcv(self.get_symbol(coin), since=1, timeframe="1d")
            if ohlcvs:
                fts = ohlcvs[0][0]
            else:
                fts = 0.0
            self.dump_first_timestamp(coin, fts)
            return fts
        elif self.exchange == "bybit":
            fts = await self.find_first_day_bybit(coin)
            return fts
        elif self.exchange == "gateio":
            ohlcvs = await self.cc.fetch_ohlcv(
                self.get_symbol(coin), since=int(date_to_ts("2018-01-01")), timeframe="1d"
            )
            if not ohlcvs:
                ohlcvs = await self.cc.fetch_ohlcv(
                    self.get_symbol(coin), since=int(date_to_ts("2020-01-01")), timeframe="1d"
                )
            if ohlcvs:
                fts = ohlcvs[0][0]
            else:
                fts = 0.0
            self.dump_first_timestamp(coin, fts)
            return fts
        elif self.exchange == "bitget":
            fts = await self.find_first_day_bitget(coin)
            return fts
        else:
            # По умолчанию
            self.dump_first_timestamp(coin, 0.0)
            return 0.0

    def load_first_timestamp(self, coin: str) -> float:
        """
        Читаем из кеша (json), какой earliest timestamp для coin.
        """
        fpath = self.cache_filepaths["first_timestamps"]
        if os.path.exists(fpath):
            try:
                ftss = json.load(open(fpath))
                if coin in ftss:
                    return ftss[coin]
            except Exception as e:
                logging.error(f"Error loading {fpath} {e}")
        return None

    def dump_first_timestamp(self, coin: str, fts: float):
        """
        Сохраняем earliest timestamp для coin в first_timestamps.json
        """
        try:
            fpath = self.cache_filepaths["first_timestamps"]
            if os.path.exists(fpath):
                try:
                    ftss = json.load(open(fpath))
                except Exception as e0:
                    logging.error(f"Error loading {fpath} {e0}")
                    ftss = {}
            else:
                make_get_filepath(fpath)
                ftss = {}

            ftss[coin] = fts
            json.dump(ftss, open(fpath, "w"), indent=True, sort_keys=True)
            if self.verbose:
                logging.info(f"{self.exchange} Dumped {fpath}")
        except Exception as e:
            logging.error(f"Error with {get_function_name()} {e}")

    # ======================================
    # ФУНКЦИИ СКАЧИВАНИЯ Binance
    # ======================================

    async def download_ohlcvs_binance(self, coin: str):
        """
        Скачивает данные из архивов binance.vision
        """
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))
        base_url = "https://data.binance.vision/data/futures/um/"
        missing_days = await self.get_missing_days_ohlcvs(coin)

        # Скопировать из старой папки, если есть
        old_dirpath = f"historical_data/ohlcvs_futures/{symbolf}/"
        if self.copy_ohlcvs_from_old_dir(dirpath, old_dirpath, missing_days, coin):
            missing_days = await self.get_missing_days_ohlcvs(coin)
            if not missing_days:
                return

        # Сперва месячные архивы
        month_now = ts_to_date_utc(utc_ms())[:7]
        missing_months = sorted({x[:7] for x in missing_days if x[:7] != month_now})
        tasks = []
        for month in missing_months:
            fpath = os.path.join(dirpath, month + ".npy")
            if not os.path.exists(fpath):
                url = f"{base_url}monthly/klines/{symbolf}/1m/{symbolf}-1m-{month}.zip"
                tasks.append(asyncio.create_task(self._download_single_binance(url, fpath)))
        await asyncio.gather(*tasks)

        # Конвертируем месячные в дневные
        for f in os.listdir(dirpath):
            if len(f) == 11:  # типа "2021-01.npy"
                npy_path = os.path.join(dirpath, f)
                df = load_ohlcv_data(npy_path)
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("datetime", inplace=True)

                daily_groups = df.groupby(df.index.date)
                n_days_dumped = 0
                for date_, daily_data in daily_groups:
                    if len(daily_data) == 1440:
                        fday = str(date_) + ".npy"
                        d_fpath = os.path.join(dirpath, fday)
                        if not os.path.exists(d_fpath):
                            n_days_dumped += 1
                            dump_ohlcv_data(daily_data, d_fpath)
                    else:
                        logging.info(
                            f"binanceusdm incomplete daily data for {coin} {date_} len={len(daily_data)}"
                        )
                if n_days_dumped:
                    logging.info(f"binanceusdm dumped {n_days_dumped} daily files for {coin} {f}")
                # Удаляем месячный файл, раз он разобран по дням
                logging.info(f"binanceusdm removing {npy_path}")
                os.remove(npy_path)

        # Теперь догружаем пропущенные дни
        missing_days = await self.get_missing_days_ohlcvs(coin)
        tasks = []
        for day in missing_days:
            fpath = os.path.join(dirpath, day + ".npy")
            if not os.path.exists(fpath):
                url = f"{base_url}daily/klines/{symbolf}/1m/{symbolf}-1m-{day}.zip"
                tasks.append(asyncio.create_task(self._download_single_binance(url, fpath)))
        await asyncio.gather(*tasks)

    async def _download_single_binance(self, url: str, fpath: str):
        """
        Скачивание одного zip архива с binance, сохранение в npy.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=240)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                content = await self._robust_fetch_url(session, url)
            # Распаковываем zip
            zbuf = BytesIO(content)
            with zipfile.ZipFile(zbuf, "r") as z:
                dfs = []
                col_names = ["timestamp", "open", "high", "low", "close", "volume"]
                for name in z.namelist():
                    with z.open(name) as f:
                        df_tmp = pd.read_csv(f, header=None)
                        df_tmp.columns = col_names + [f"extra_{i}" for i in range(len(df_tmp.columns) - len(col_names))]
                        dfs.append(df_tmp[col_names])
                if not dfs:
                    return
                dfc = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
                dfc = dfc[dfc.timestamp != "open_time"].astype(float)
                if not dfc.empty:
                    dfc = ensure_millis(dfc)
                    dump_ohlcv_data(dfc, fpath)
                    if self.verbose:
                        logging.info(f"binanceusdm Dumped data {fpath}")
        except Exception as e:
            logging.error(f"binanceusdm Failed to download {url}: {e}")
            traceback.print_exc()

    # ======================================
    # СКАЧИВАНИЕ Bybit
    # ======================================
    async def download_ohlcvs_bybit(self, coin: str):
        missing_days = await self.get_missing_days_ohlcvs(coin)
        if not missing_days:
            return
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))
        old_dirpath = f"historical_data/ohlcvs_bybit/{symbolf}/"

        if self.copy_ohlcvs_from_old_dir(dirpath, old_dirpath, missing_days, coin):
            missing_days = await self.get_missing_days_ohlcvs(coin)
            if not missing_days:
                return

        base_url = "https://public.bybit.com/trading/"
        webpage = urlopen(f"{base_url}{symbolf}/").read().decode()

        filenames = [f"{symbolf}{day}.csv.gz" for day in missing_days if f"{symbolf}{day}.csv.gz" in webpage]
        if not filenames:
            return

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as session:
            tasks = []
            for fn in filenames:
                url = f"{base_url}{symbolf}/{fn}"
                day_str = fn[-17:-7]
                tasks.append(asyncio.create_task(self._download_single_bybit(session, url, dirpath, day_str)))
            await asyncio.gather(*tasks)

    async def _download_single_bybit(self, session, url: str, dirpath: str, day: str):
        """
        Скачиваем сырые трейды Bybit (csv.gz), конвертируем их в 1m OHLCV.
        """
        try:
            content = await self._robust_fetch_url(session, url)
            with gzip.open(BytesIO(content)) as gz:
                raw = pd.read_csv(gz)
            # группируем по минутным интервалам
            interval = 60000
            groups = raw.groupby((raw.timestamp * 1000) // interval * interval)
            ohlcvs = pd.DataFrame({
                "open": groups.price.first(),
                "high": groups.price.max(),
                "low": groups.price.min(),
                "close": groups.price.last(),
                "volume": groups["size"].sum(),
            })
            ohlcvs["timestamp"] = ohlcvs.index
            fpath = os.path.join(dirpath, day + ".npy")
            dump_ohlcv_data(ensure_millis(ohlcvs[["timestamp", "open", "high", "low", "close", "volume"]]), fpath)
            if self.verbose:
                logging.info(f"bybit Dumped {fpath}")
        except Exception as e:
            logging.error(f"bybit error {url}: {e}")
            traceback.print_exc()

    async def find_first_day_bybit(self, coin: str, webpage=None) -> float:
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        base_url = "https://public.bybit.com/trading/"
        if webpage is None:
            webpage = urlopen(f"{base_url}{symbolf}/").read().decode()

        # Ищем все строки формата "xxxx-xx-xx.csv.gz"
        dates = []
        for x in webpage.split(".csv.gz"):
            d = x[-10:]
            if is_valid_date(d):
                dates.append(d)
        if not dates:
            self.dump_first_timestamp(coin, 0.0)
            return 0.0

        earliest_day = sorted(dates)[0]
        fts = date_to_ts(earliest_day)
        self.dump_first_timestamp(coin, fts)
        return fts

    # ======================================
    # СКАЧИВАНИЕ Bitget
    # ======================================
    async def download_ohlcvs_bitget(self, coin: str):
        fts = await self.find_first_day_bitget(coin)
        if fts == 0.0:
            return
        missing_days = await self.get_missing_days_ohlcvs(coin)
        if not missing_days:
            return
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        if not symbolf:
            return

        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))
        base_url = "https://img.bitgetimg.com/online/kline/"
        tasks = []
        for day in missing_days:
            fpath = os.path.join(dirpath, day + ".npy")
            url = self._get_url_bitget(base_url, symbolf, day)
            tasks.append(asyncio.create_task(self._download_single_bitget(url, fpath)))
        await asyncio.gather(*tasks)

    def _get_url_bitget(self, base_url, symbolf, day: str):
        if day <= "2024-04-18":
            return f"{base_url}{symbolf}/{symbolf}_UMCBL_1min_{day.replace('-', '')}.zip"
        else:
            return f"{base_url}{symbolf}/UMCBL/{day.replace('-', '')}.zip"

    async def _download_single_bitget(self, url: str, fpath: str):
        """
        Скачиваем архив Bitget, распаковываем Excel-файл, конвертируем в npy.
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as session:
                content = await self._robust_fetch_url(session, url)
            with zipfile.ZipFile(BytesIO(content), "r") as zf:
                col_names = ["timestamp", "open", "high", "low", "close", "volume"]
                dfs = []
                for n in zf.namelist():
                    with zf.open(n) as file_ref:
                        df_tmp = ensure_millis(pd.read_excel(file_ref))
                        df_tmp.columns = col_names + [f"extra_{i}" for i in range(len(df_tmp.columns) - len(col_names))]
                        dfs.append(df_tmp[col_names])
                if not dfs:
                    return
                df = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
                df = df[df.timestamp != "open_time"]
                df = ensure_millis(df)
                dump_ohlcv_data(df, fpath)
                if self.verbose:
                    logging.info(f"bitget Dumped daily data {fpath}")
        except Exception as e:
            logging.error(f"bitget error {url}: {e}")

    async def find_first_day_bitget(self, coin: str, start_year=2020) -> float:
        """
        Пробуем бинарный поиск по датам, чтобы найти первый день, для которого есть архив.
        """
        if (fts := self.load_first_timestamp(coin)):
            return fts
        if not self.markets:
            await self.load_markets()
        symbol = self.get_symbol(coin).replace("/USDT:", "")
        if not symbol:
            self.dump_first_timestamp(coin, 0.0)
            return 0.0

        base_url = "https://img.bitgetimg.com/online/kline/"
        start = datetime.datetime(start_year, 1, 1)
        end = datetime.datetime.now()
        earliest = None

        while start <= end:
            mid = start + (end - start) // 2
            date_str = mid.strftime("%Y%m%d")
            url = self._get_url_bitget(base_url, symbol, mid.strftime("%Y-%m-%d"))

            try:
                await self.check_rate_limit()
                async with aiohttp.ClientSession() as session:
                    async with session.head(url) as resp:
                        if self.verbose:
                            logging.info(
                                f"bitget, searching first data for {symbol} {str(mid)[:10]}, resp={resp.status}"
                            )
                        if resp.status == 200:
                            earliest = mid
                            end = mid - datetime.timedelta(days=1)
                        else:
                            start = mid + datetime.timedelta(days=1)
            except Exception as e:
                start = mid + datetime.timedelta(days=1)

        if earliest:
            # Проверим предыдущий день
            prev_day = earliest - datetime.timedelta(days=1)
            prev_url = self._get_url_bitget(base_url, symbol, prev_day.strftime("%Y-%m-%d"))
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.head(prev_url) as response:
                        if response.status == 200:
                            earliest = prev_day
            except Exception:
                pass
            if self.verbose:
                logging.info(f"Bitget, found first day for {symbol}: {earliest.strftime('%Y-%m-%d')}")
            fts = date_to_ts(earliest.strftime("%Y-%m-%d"))
            self.dump_first_timestamp(coin, fts)
            return fts
        return 0.0

    # ======================================
    # СКАЧИВАНИЕ GateIO
    # ======================================
    async def download_ohlcvs_gateio(self, coin: str):
        """
        GateIO не имеет публичных zip-архивов, поэтому скачиваем через REST.
        Для каждого пропущенного дня делаем fetch_ohlcv(...).
        """
        missing_days = await self.get_missing_days_ohlcvs(coin)
        if not missing_days:
            return
        if self.cc is None:
            self.load_cc()

        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))
        symbol = self.get_symbol(coin)

        tasks = []
        for day in missing_days:
            tasks.append(asyncio.create_task(self._fetch_and_save_day_gateio(symbol, day, dirpath)))
        await asyncio.gather(*tasks)

    async def _fetch_and_save_day_gateio(self, symbol: str, day: str, dirpath: str):
        """
        Одноразовый запрос на gateio для 1 дня (limit ~1500 минут).
        """
        await self.check_rate_limit()

        fpath = os.path.join(dirpath, f"{day}.npy")
        start_ts_day = date_to_ts(day)  # полночь UTC
        end_ts_day = start_ts_day + 24 * 60 * 60 * 1000
        interval = "1m"
        limit = 1500
        try:
            ohlcvs = await self.cc.fetch_ohlcv(symbol, timeframe=interval, since=start_ts_day, limit=limit)
        except Exception as e:
            logging.error(f"gateio: error for {symbol} {day} {e}")
            return

        if not ohlcvs:
            if self.verbose:
                logging.info(f"No data returned for GateIO {symbol} {day}")
            return

        df_day = pd.DataFrame(ohlcvs, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_day = df_day[(df_day.timestamp >= start_ts_day) & (df_day.timestamp < end_ts_day)].reset_index(drop=True)

        # Предположим, что volume в df_day уже quote-volume
        # Если нужно сделать base-volume, можно пересчитать: df_day["volume"] /= df_day["close"]
        if len(df_day) == 1440:
            dump_ohlcv_data(ensure_millis(df_day), fpath)
            if self.verbose:
                logging.info(f"gateio Dumped daily OHLCV data for {symbol} to {fpath}")


# ======================================
# ФУНКЦИИ ДЛЯ ОБЩЕЙ ЛОГИКИ (prepare_hlcvs и т.д.)
# ======================================

async def prepare_hlcvs(config: dict, exchange: str):
    coins = sorted(
        set([symbol_to_coin(c) for c in config["live"]["approved_coins"]["long"]])
        | set([symbol_to_coin(c) for c in config["live"]["approved_coins"]["short"]])
    )
    if exchange == "binance":
        exchange = "binanceusdm"
    start_date = config["backtest"]["start_date"]
    end_date = format_end_date(config["backtest"]["end_date"])

    om = OHLCVManager(
        exchange,
        start_date,
        end_date,
        gap_tolerance_ohlcvs_minutes=config["backtest"]["gap_tolerance_ohlcvs_minutes"],
    )

    try:
        mss, timestamps, hlcvs = await prepare_hlcvs_internal(
            config, coins, exchange, start_date, end_date, om
        )

        om.update_date_range(timestamps[0], timestamps[-1])
        btc_df = await om.get_ohlcvs("BTC")
        if btc_df.empty:
            raise ValueError(f"Failed to fetch BTC/USD prices from {exchange}")

        btc_df = btc_df.set_index("timestamp").reindex(timestamps, method="ffill").reset_index()
        btc_usd_prices = btc_df["close"].values
        return mss, timestamps, hlcvs, btc_usd_prices
    finally:
        if om.cc:
            await om.cc.close()


async def prepare_hlcvs_internal(config, coins, exchange, start_date, end_date, om):
    end_ts = date_to_ts(end_date)
    minimum_coin_age_days = config["live"]["minimum_coin_age_days"]
    interval_ms = 60000

    first_timestamps_unified = await get_first_timestamps_unified(coins)
    cache_dir = Path(f"./caches/hlcvs_data/{uuid4().hex[:16]}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    valid_coins = {}
    global_start_time = float("inf")
    global_end_time = float("-inf")
    await om.load_markets()
    min_coin_age_ms = 1000 * 60 * 60 * 24 * minimum_coin_age_days

    for coin in coins:
        adjusted_start_ts = date_to_ts(start_date)
        if not om.has_coin(coin):
            logging.info(f"{exchange} coin {coin} missing, skipping")
            continue
        if coin not in first_timestamps_unified:
            logging.info(f"coin {coin} missing from first_timestamps_unified, skipping")
            continue
        if minimum_coin_age_days > 0.0:
            first_ts = await om.get_first_timestamp(coin)
            if first_ts >= end_ts:
                logging.info(f"{exchange} Coin {coin} too young, start {ts_to_date_utc(first_ts)}. Skipping")
                continue
            first_ts_plus_min_coin_age = first_timestamps_unified[coin] + min_coin_age_ms
            if first_ts_plus_min_coin_age >= end_ts:
                logging.info(
                    f"{exchange} Coin {coin} not traded due to min_coin_age {int(minimum_coin_age_days)} days"
                )
                continue
            new_adjusted_start_ts = max(first_timestamps_unified[coin] + min_coin_age_ms, first_ts)
            if new_adjusted_start_ts > adjusted_start_ts:
                logging.info(
                    f"{exchange} Coin {coin}: Adjusting start date from {start_date} "
                    f"to {ts_to_date_utc(new_adjusted_start_ts)}"
                )
                adjusted_start_ts = new_adjusted_start_ts
        try:
            om.update_date_range(adjusted_start_ts)
            df = await om.get_ohlcvs(coin)
            data = df[["timestamp", "high", "low", "close", "volume"]].values
        except Exception as e:
            logging.error(f"error with get_ohlcvs for {coin} {e}. Skipping")
            traceback.print_exc()
            continue
        if len(data) == 0:
            continue

        # Проверяем, что нет пропусков по минутам
        if not (np.diff(data[:, 0]) == interval_ms).all():
            raise ValueError(f"Gaps in hlcv data {coin}")

        file_path = cache_dir / f"{coin}.npy"
        dump_ohlcv_data(data, file_path)

        valid_coins[coin] = file_path
        global_start_time = min(global_start_time, data[0, 0])
        global_end_time = max(global_end_time, data[-1, 0])

    if not valid_coins:
        raise ValueError("No valid coins found with data")

    n_timesteps = int((global_end_time - global_start_time) / interval_ms) + 1
    timestamps = np.arange(global_start_time, global_end_time + interval_ms, interval_ms)
    n_coins = len(valid_coins)
    unified_array = np.zeros((n_timesteps, n_coins, 4))

    logging.info(f"{exchange} Unifying data for {len(valid_coins)} coins into one array...")

    for i, coin in enumerate(tqdm(valid_coins, desc="Processing coins", unit="coin")):
        ohlcv = np.load(valid_coins[coin])
        start_idx = int((ohlcv[0, 0] - global_start_time) / interval_ms)
        end_idx = start_idx + len(ohlcv)
        coin_data = ohlcv[:, 1:]
        unified_array[start_idx:end_idx, i, :] = coin_data

        if start_idx > 0:
            unified_array[:start_idx, i, :3] = coin_data[0, 2]
        if end_idx < n_timesteps:
            unified_array[end_idx:, i, :3] = coin_data[-1, 2]
        os.remove(valid_coins[coin])

    try:
        os.rmdir(cache_dir)
    except OSError:
        pass

    mss = {coin: om.get_market_specific_settings(coin) for coin in sorted(valid_coins)}
    return mss, timestamps, unified_array


# --------------------------------------------------------
# prepare_hlcvs_combined() - аналогичная логика, объединяющая несколько бирж
# --------------------------------------------------------

async def prepare_hlcvs_combined(config):
    exchanges_to_consider = [
        "binanceusdm" if e == "binance" else e for e in config["backtest"]["exchanges"]
    ]
    om_dict = {}
    for ex in exchanges_to_consider:
        om_dict[ex] = OHLCVManager(
            ex,
            config["backtest"]["start_date"],
            config["backtest"]["end_date"],
            gap_tolerance_ohlcvs_minutes=config["backtest"]["gap_tolerance_ohlcvs_minutes"],
        )
    btc_om = None

    try:
        mss, timestamps, unified_array = await _prepare_hlcvs_combined_impl(config, om_dict)
        # Заодно берём btc/usd
        btc_exchange = exchanges_to_consider[0] if len(exchanges_to_consider) == 1 else "binanceusdm"
        btc_om = OHLCVManager(
            btc_exchange,
            config["backtest"]["start_date"],
            config["backtest"]["end_date"],
            gap_tolerance_ohlcvs_minutes=config["backtest"]["gap_tolerance_ohlcvs_minutes"],
        )
        btc_df = await btc_om.get_ohlcvs("BTC")
        if btc_df.empty:
            raise ValueError(f"Failed to fetch BTC/USD prices from {btc_exchange}")

        btc_df = btc_df.set_index("timestamp").reindex(timestamps, method="ffill").reset_index()
        btc_usd_prices = btc_df["close"].values
        return mss, timestamps, unified_array, btc_usd_prices
    finally:
        for om in om_dict.values():
            if om.cc:
                await om.cc.close()
        if btc_om and btc_om.cc:
            await btc_om.cc.close()


async def _prepare_hlcvs_combined_impl(config, om_dict):
    """
    Амальгамация данных нескольких бирж.
    Здесь оставим логику без особых изменений, просто обращаем внимание, 
    что внутри уже есть вызовы om.get_ohlcvs(coin) – теперь они надёжнее.
    """
    # (Логика выбора лучшей биржи для каждой монеты, объединение в один массив и т.д.)
    # Оставляем код из исходного скрипта без сильных изменений.
    start_date = config["backtest"]["start_date"]
    end_date = format_end_date(config["backtest"]["end_date"])
    start_ts = date_to_ts(start_date)
    end_ts = date_to_ts(end_date)
    coins = sorted(
        set([symbol_to_coin(c) for c in config["live"]["approved_coins"]["long"]])
        | set([symbol_to_coin(c) for c in config["live"]["approved_coins"]["short"]])
    )
    exchanges_to_consider = list(om_dict.keys())

    min_coin_age_days = config["live"].get("minimum_coin_age_days", 0.0)
    min_coin_age_ms = int(min_coin_age_days * 24 * 60 * 60 * 1000)
    first_timestamps_unified = await get_first_timestamps_unified(coins)

    for ex in exchanges_to_consider:
        await om_dict[ex].load_markets()

    chosen_data_per_coin = {}
    chosen_mss_per_coin = {}

    # Собираем данные от всех бирж и выбираем лучшую
    tasks_for_all_coins = []
    for coin in coins:
        coin_fts = first_timestamps_unified.get(coin, 0.0)
        if coin_fts == 0.0:
            logging.info(f"Skipping coin {coin}, no first timestamp recorded.")
            continue
        if coin_fts + min_coin_age_ms >= end_ts:
            logging.info(
                f"Skipping coin {coin}: min_coin_age_days = {min_coin_age_days}"
            )
            continue
        effective_start_ts = max(start_ts, coin_fts + min_coin_age_ms)
        if effective_start_ts >= end_ts:
            continue

        # Создаём таски на скачку с каждой биржи (fetch_data_for_coin_and_exchange).
        tasks = []
        for ex in exchanges_to_consider:
            tasks.append(
                asyncio.create_task(
                    fetch_data_for_coin_and_exchange(
                        coin, ex, om_dict[ex], effective_start_ts, end_ts
                    )
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        exchange_candidates = []
        for r in results:
            if r is None or isinstance(r, Exception):
                continue
            exN, df, coverage_count, gap_count, total_volume = r
            exchange_candidates.append((exN, df, coverage_count, gap_count, total_volume))

        if not exchange_candidates:
            logging.info(f"No exchange data found at all for coin {coin}. Skipping.")
            continue

        # Сортируем по coverage_count (desc), gap_count (asc), volume (desc)
        if len(exchange_candidates) == 1:
            best_exchange, best_df, best_cov, best_gaps, best_vol = exchange_candidates[0]
        else:
            exchange_candidates.sort(key=lambda x: (x[2], -x[3], x[4]), reverse=True)
            best_exchange, best_df, best_cov, best_gaps, best_vol = exchange_candidates[0]

        logging.info(f"{coin} exchange preference: {[x[0] for x in exchange_candidates]}")
        chosen_data_per_coin[coin] = best_df
        chosen_mss_per_coin[coin] = om_dict[best_exchange].get_market_specific_settings(coin)
        chosen_mss_per_coin[coin]["exchange"] = best_exchange

    if not chosen_data_per_coin:
        raise ValueError("No coin data found on any exchange for the requested date range.")

    # Объединяем всё в единый 3D-массив (n_timestamps, n_coins, 4)
    global_start_time = min(df.timestamp.iloc[0] for df in chosen_data_per_coin.values())
    global_end_time = max(df.timestamp.iloc[-1] for df in chosen_data_per_coin.values())
    timestamps = np.arange(global_start_time, global_end_time + 60000, 60000)
    n_timesteps = len(timestamps)
    valid_coins = sorted(chosen_data_per_coin.keys())
    n_coins = len(valid_coins)

    # (Можно сделать логику для volume_ratios, при желании — здесь из исходного кода)
    # ...

    unified_array = np.zeros((n_timesteps, n_coins, 4), dtype=np.float64)
    for i, coin in enumerate(valid_coins):
        df = chosen_data_per_coin[coin].copy()
        df = df.set_index("timestamp").reindex(timestamps)
        df["close"] = df["close"].ffill().bfill()
        for col in ["open", "high", "low"]:
            df[col] = df[col].fillna(df["close"])
        df["volume"] = df["volume"].fillna(0.0)

        coin_data = df[["high", "low", "close", "volume"]].values
        unified_array[:, i, :] = coin_data

    return chosen_mss_per_coin, timestamps, unified_array


async def fetch_data_for_coin_and_exchange(
    coin: str, ex: str, om: OHLCVManager, effective_start_ts: int, end_ts: int
):
    """
    Вспомогательная функция: скачивает 1m-данные для монеты coin на бирже ex в интервале [effective_start_ts, end_ts].
    Возвращает кортеж (ex, df, coverage_count, gap_count, total_volume) 
    или None, если данных нет/ошибка.
    """
    if not om.has_coin(coin):
        return None

    om.update_date_range(effective_start_ts, end_ts)
    try:
        df = await om.get_ohlcvs(coin)
    except Exception as e:
        logging.warning(f"Error retrieving {coin} from {ex}: {e}")
        return None
    if df.empty:
        return None

    df = df[(df.timestamp >= effective_start_ts) & (df.timestamp <= end_ts)].reset_index(drop=True)
    if df.empty:
        return None

    coverage_count = len(df)
    intervals = np.diff(df["timestamp"].values)
    gap_count = sum((gap // 60000) - 1 for gap in intervals if gap > 60000)
    total_volume = df["volume"].sum()
    return (ex, df, coverage_count, gap_count, total_volume)


async def compute_exchange_volume_ratios(
    exchanges: List[str],
    coins: List[str],
    start_date: str,
    end_date: str,
    om_dict: Dict[str, OHLCVManager] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Пример функции для сравнения средних объёмов по парам бирж (ex0, ex1).
    Возвращает словарь {(ex0, ex1): ratio} и т.д.
    """
    if om_dict is None:
        om_dict = {ex: OHLCVManager(ex, start_date, end_date) for ex in exchanges}
        await asyncio.gather(*[om_dict[ex].load_markets() for ex in om_dict])

    exchange_pairs = []
    for i, ex0 in enumerate(sorted(exchanges)):
        for ex1 in exchanges[i + 1 :]:
            exchange_pairs.append((ex0, ex1))

    all_data = {}
    for coin in coins:
        # Убедимся, что coin есть на всех биржах
        if not all(om_dict[ex].has_coin(coin) for ex in exchanges):
            continue

        tasks = []
        for ex in exchanges:
            om = om_dict[ex]
            om.update_date_range(start_date, end_date)
            tasks.append(om.get_ohlcvs(coin))
        dfs = await asyncio.gather(*tasks, return_exceptions=True)
        # приводим к одному формату
        for i, df in enumerate(dfs):
            if isinstance(df, Exception) or df is None or df.empty:
                dfs[i] = pd.DataFrame()

        if any(df.empty for df in dfs):
            continue

        daily_volumes = []
        for df in dfs:
            df["day"] = df["timestamp"] // 86400000
            grouped = df.groupby("day", as_index=False)["volume"].sum()
            day_dict = dict(zip(grouped["day"], grouped["volume"]))
            daily_volumes.append(day_dict)

        sets_of_days = [set(dv.keys()) for dv in daily_volumes]
        common_days = set.intersection(*sets_of_days)
        if not common_days:
            continue

        coin_data = {}
        for ex0, ex1 in exchange_pairs:
            i0 = exchanges.index(ex0)
            i1 = exchanges.index(ex1)
            sum0 = sum(daily_volumes[i0][d] for d in common_days)
            sum1 = sum(daily_volumes[i1][d] for d in common_days)
            ratio = sum0 / sum1 if sum1 > 0 else 0.0
            coin_data[(ex0, ex1)] = ratio

        if coin_data:
            all_data[coin] = coin_data

    averages = {}
    if not all_data:
        return averages

    used_pairs = set()
    for coin in all_data:
        for pair in all_data[coin]:
            used_pairs.add(pair)

    for pair in used_pairs:
        ratios_for_pair = []
        for coin in all_data:
            if pair in all_data[coin]:
                ratios_for_pair.append(all_data[coin][pair])
        if ratios_for_pair:
            averages[pair] = float(np.mean(ratios_for_pair))
        else:
            averages[pair] = 0.0
    return averages


# ======================================
# Основная точка входа
# ======================================
async def add_all_eligible_coins_to_config(config):
    """
    Если в конфиге указано empty_means_all_approved, тогда пытаемся загрузить все монеты.
    """
    path = config["live"]["approved_coins"]
    if config["live"]["empty_means_all_approved"] and path in [
        [""], [], None, "", 0, 0.0, {"long": [], "short": []}, {"long": [""], "short": [""]},
    ]:
        approved_coins = await get_all_eligible_coins(config["backtest"]["exchanges"])
        config["live"]["approved_coins"] = {"long": approved_coins, "short": approved_coins}


async def get_all_eligible_coins(exchanges):
    """
    Собрать все монеты (coins), которые есть на всех указанных биржах, используя OHLCVManager + load_markets.
    """
    oms = {}
    for ex in exchanges:
        oms[ex] = OHLCVManager(ex, verbose=False)
    await asyncio.gather(*[oms[ex].load_markets() for ex in oms])
    approved_coins = set()
    for ex in oms:
        for s in oms[ex].markets:
            if oms[ex].has_coin(s):
                coin = symbol_to_coin(s)
                if coin:
                    approved_coins.add(coin)
    return sorted(approved_coins)


async def main():
    parser = argparse.ArgumentParser(prog="downloader", description="download ohlcv data")
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    template_config = get_template_live_config("v7")
    del template_config["optimize"]
    del template_config["bot"]
    template_config["live"] = {
        k: v
        for k, v in template_config["live"].items()
        if k in {"approved_coins", "ignored_coins"}
    }
    template_config["backtest"] = {
        k: v
        for k, v in template_config["backtest"].items()
        if k in {"combine_ohlcvs", "end_date", "start_date", "exchanges"}
    }
    add_arguments_recursively(parser, template_config)
    args = parser.parse_args()

    if args.config_path is None:
        logging.info("loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=False)
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path)

    await add_all_eligible_coins_to_config(config)

    oms = {}
    try:
        for ex in config["backtest"]["exchanges"]:
            oms[ex] = OHLCVManager(
                ex, config["backtest"]["start_date"], config["backtest"]["end_date"]
            )
        logging.info(f"loading markets for {config['backtest']['exchanges']}")
        await asyncio.gather(*[oms[ex].load_markets() for ex in oms])

        coins = [x for y in config["live"]["approved_coins"].values() for x in y]
        for coin in sorted(set(coins)):
            tasks = {}
            for ex in oms:
                try:
                    tasks[ex] = asyncio.create_task(oms[ex].get_ohlcvs(coin))
                except Exception as e:
                    logging.error(f"{ex} {coin} error (a) with get_ohlcvs() {e}")

            # дожидаемся результатов
            for ex in tasks:
                try:
                    await tasks[ex]
                except Exception as e:
                    logging.error(f"{ex} {coin} error (b) with get_ohlcvs() {e}")

    finally:
        for om in oms.values():
            if om.cc:
                await om.cc.close()


if __name__ == "__main__":
    asyncio.run(main())
