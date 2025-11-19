import re
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def parse_price(price_str: str) -> Optional[float]:
    if not isinstance(price_str, str): return None
    # Remove currency symbols and commas
    cleaned = re.sub(r"[^\d.]", "", str(price_str))
    try: return float(cleaned)
    except: return None

def parse_ram(ram_val) -> Optional[int]:
    if isinstance(ram_val, (int, float)): return int(ram_val)
    if not isinstance(ram_val, str): return None
    match = re.search(r'(\d+)', ram_val)
    return int(match.group(1)) if match else None

def parse_weight(w) -> Optional[float]:
    if isinstance(w, (int, float)): return float(w)
    if not isinstance(w, str): return None
    # Handle "1.5 kg" or "1500 g"
    match = re.search(r'(\d+(\.\d+)?)', w)
    return float(match.group(1)) if match else None

def parse_os(os_str) -> str:
    s = str(os_str).lower()
    if 'windows' in s: return 'Windows'
    if 'mac' in s: return 'macOS'
    if 'linux' in s or 'ubuntu' in s: return 'Linux'
    return 'Other'

def create_heuristic_features(row: pd.Series) -> pd.Series:
    # Gaming Heuristic
    row['is_gaming'] = 0
    gpu_t = str(row.get('gpu_type', '')).lower()
    gpu_b = str(row.get('gpu_brand', '')).lower()
    gpu_m = str(row.get('gpu_model', '')).lower()
    
    if 'discrete' in gpu_t or 'nvidia' in gpu_b or 'rtx' in gpu_m or 'gtx' in gpu_m:
        row['is_gaming'] = 1
        
    # Ultrabook Heuristic
    row['is_ultrabook'] = 0
    w = row.get('weight_kg', 2.0)
    if w <= 1.5 and 'ssd' in str(row.get('storage_type','')).lower():
        row['is_ultrabook'] = 1
    return row

def calculate_cpu_score(row: pd.Series) -> int:
    score = 0
    series = str(row.get('cpu_series', '')).lower()
    if 'i9' in series or 'ryzen 9' in series or 'm2 max' in series: score = 10
    elif 'i7' in series or 'ryzen 7' in series or 'm2 pro' in series: score = 8
    elif 'i5' in series or 'ryzen 5' in series or 'm2' in series: score = 6
    elif 'i3' in series or 'ryzen 3' in series: score = 4
    else: score = 2
    return score