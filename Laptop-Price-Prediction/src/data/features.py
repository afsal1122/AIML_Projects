# File: src/data/features.py
"""
Feature Engineering Functions.

This module contains all the parsing and transformation functions
used to convert raw text data into structured features for the model.
"""

import re
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def parse_price(price_str: str) -> Optional[float]:
    """Converts a raw price string (e.g., '₹74,990', '$1,299') to a numeric INR value."""
    if not isinstance(price_str, str):
        return None
    
    price_str = price_str.strip()
    
    # Basic currency conversion (illustrative, use an API for real rates)
    multiplier = 1.0
    if '$' in price_str:
        multiplier = 83.0  # Example rate
    elif '€' in price_str:
        multiplier = 90.0 # Example rate

    # Remove currency symbols, commas, and other noise
    cleaned_price = re.sub(r"[₹,$\s€]", "", price_str)
    
    try:
        price = float(cleaned_price)
        return price * multiplier
    except (ValueError, TypeError):
        return None

def parse_ram(ram_str: str) -> Optional[int]:
    """Extracts RAM in GB from a string (e.g., '16 GB LPDDR5')."""
    if not isinstance(ram_str, str):
        return None
    
    match = re.search(r'(\d+)\s*GB', ram_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def parse_storage(storage_str: str) -> Tuple[Optional[int], str]:
    """
    Extracts storage capacity (in GB) and type (SSD, HDD, Hybrid, Unknown).
    e.g., '512 GB SSD' -> (512, 'SSD')
    e.g., '1 TB HDD' -> (1024, 'HDD')
    e.g., '1 TB HDD + 256 GB SSD' -> (1280, 'Hybrid')
    """
    if not isinstance(storage_str, str):
        return None, 'Unknown'

    storage_str = storage_str.lower()
    total_gb = 0
    storage_types = set()

    # Find TB values
    tb_matches = re.findall(r'(\d+)\s*TB', storage_str, re.IGNORECASE)
    for tb in tb_matches:
        total_gb += int(tb) * 1024

    # Find GB values
    gb_matches = re.findall(r'(\d+)\s*GB', storage_str, re.IGNORECASE)
    for gb in gb_matches:
        total_gb += int(gb)

    if 'ssd' in storage_str:
        storage_types.add('SSD')
    if 'hdd' in storage_str or 'hard drive' in storage_str:
        storage_types.add('HDD')

    if not storage_types and total_gb > 0:
        # Guess based on common configs
        if 'emmc' in storage_str:
            storage_types.add('Other')
        else:
            storage_types.add('SSD') # Default assumption

    # Determine final type
    if len(storage_types) > 1:
        storage_type = 'Hybrid'
    elif 'SSD' in storage_types:
        storage_type = 'SSD'
    elif 'HDD' in storage_types:
        storage_type = 'HDD'
    else:
        storage_type = 'Unknown'
        
    return total_gb if total_gb > 0 else None, storage_type

def parse_cpu(cpu_str: str) -> Tuple[str, str, Optional[int]]:
    """
    Extracts CPU brand, series, and generation.
    e.g., 'Intel Core i7 12th Gen' -> ('Intel', 'Core i7', 12)
    e.g., 'Apple M2 Pro' -> ('Apple', 'M2 Pro', 0)
    e.g., 'AMD Ryzen 5 5600H' -> ('AMD', 'Ryzen 5', 5)
    """
    if not isinstance(cpu_str, str):
        return 'Unknown', 'Unknown', None
    
    cpu_str_low = cpu_str.lower()
    
    # Brand
    if 'intel' in cpu_str_low:
        brand = 'Intel'
    elif 'amd' in cpu_str_low:
        brand = 'AMD'
    elif 'apple' in cpu_str_low:
        brand = 'Apple'
    else:
        brand = 'Other'

    # Series
    series = 'Unknown'
    if brand == 'Intel':
        for s in ['Core i9', 'Core i7', 'Core i5', 'Core i3', 'Pentium', 'Celeron']:
            if s.lower() in cpu_str_low:
                series = s
                break
    elif brand == 'AMD':
        for s in ['Ryzen 9', 'Ryzen 7', 'Ryzen 5', 'Ryzen 3', 'Athlon']:
            if s.lower() in cpu_str_low:
                series = s
                break
    elif brand == 'Apple':
        for s in ['M2 Pro', 'M2 Max', 'M2', 'M1 Pro', 'M1 Max', 'M1']:
            if s in cpu_str: # Case sensitive for Apple
                series = s
                break
    
    # Generation (simplified)
    generation = None
    if brand == 'Intel':
        # e.g., "12th Gen" or "i7-12700H"
        match = re.search(r'(\d{2})th\s*Gen', cpu_str, re.IGNORECASE)
        if match:
            generation = int(match.group(1))
        else:
            match = re.search(r'i[3579]-(\d{1,2})', cpu_str, re.IGNORECASE)
            if match:
                gen_num = int(match.group(1))
                if gen_num > 5: # Simple heuristic
                    generation = gen_num if gen_num <= 14 else int(str(gen_num)[0:2])
                
    elif brand == 'AMD':
        # e.g., "Ryzen 5 5600H" -> 5th gen
        match = re.search(r'Ryzen\s*[3579]\s*(\d)\d{3}', cpu_str, re.IGNORECASE)
        if match:
            generation = int(match.group(1))

    elif brand == 'Apple':
        generation = 0 # Use 0 as a placeholder
        
    return brand, series, generation

def parse_gpu(gpu_str: str) -> Tuple[str, str, str, Optional[int]]:
    """
    Extracts GPU brand, model, type (Discrete/Integrated), and VRAM.
    e.g., 'NVIDIA GeForce RTX 3060 6GB' -> ('NVIDIA', 'RTX 3060', 'Discrete', 6)
    e.g., 'Intel Iris Xe' -> ('Intel', 'Iris Xe', 'Integrated', None)
    """
    if not isinstance(gpu_str, str):
        return 'Unknown', 'Unknown', 'Unknown', None
        
    gpu_str_low = gpu_str.lower()
    
    # Brand and Type
    if 'nvidia' in gpu_str_low:
        brand = 'NVIDIA'
        gpu_type = 'Discrete'
    elif 'amd' in gpu_str_low and ('radeon' in gpu_str_low and 'vega' not in gpu_str_low):
        brand = 'AMD'
        gpu_type = 'Discrete'
    elif 'intel' in gpu_str_low:
        brand = 'Intel'
        gpu_type = 'Integrated'
    elif 'apple' in gpu_str_low:
        brand = 'Apple'
        gpu_type = 'Integrated'
    else:
        brand = 'Unknown'
        gpu_type = 'Integrated' if 'integrated' in gpu_str_low else 'Unknown'

    # Model (simplified)
    model = 'Unknown'
    if brand == 'NVIDIA':
        match = re.search(r'(RTX\s*\d{4}|GTX\s*\d{4}|MX\d{3})', gpu_str, re.IGNORECASE)
        if match:
            model = match.group(1).upper()
    elif brand == 'AMD' and gpu_type == 'Discrete':
        match = re.search(r'(RX\s*\d{4})', gpu_str, re.IGNORECASE)
        if match:
            model = match.group(1).upper()
    elif brand == 'Intel':
        if 'iris' in gpu_str_low:
            model = 'Iris Xe'
        elif 'uhd' in gpu_str_low:
            model = 'Intel UHD'
            
    # VRAM
    vram = None
    match = re.search(r'(\d+)\s*GB', gpu_str, re.IGNORECASE)
    if match and gpu_type == 'Discrete':
        vram = int(match.group(1))

    return brand, model, gpu_type, vram

def parse_display(display_str: str, res_str: str) -> Tuple[Optional[float], str, Optional[float]]:
    """
    Parses display size, resolution, and calculates PPI.
    """
    size_in = None
    if isinstance(display_str, str):
        match = re.search(r'(\d{2}(\.\d)?)', display_str)
        if match:
            size_in = float(match.group(1))

    resolution = 'Unknown'
    if isinstance(res_str, str):
        match = re.search(r'(\d{4}\s*x\s*\d{4})', res_str)
        if match:
            resolution = match.group(1).replace(" ", "")

    ppi = calculate_ppi(size_in, resolution)
    return size_in, resolution, ppi

def calculate_ppi(screen_size_in: Optional[float], resolution: str) -> Optional[float]:
    """Calculates Pixels Per Inch (PPI)."""
    if not screen_size_in or not resolution or 'x' not in resolution:
        return None
    try:
        width, height = map(int, resolution.split('x'))
        ppi = (np.sqrt(width**2 + height**2) / screen_size_in)
        return round(ppi, 2)
    except (ValueError, TypeError):
        return None

def parse_weight(weight_str: str) -> Optional[float]:
    """Extracts weight in kg."""
    if not isinstance(weight_str, str):
        return None
    
    # Look for 'kg' first
    match = re.search(r'(\d+(\.\d+)?)\s*kg', weight_str, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Look for 'g' or 'grams' and convert
    match = re.search(r'(\d+)\s*(g|grams)', weight_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 1000.0
    
    return None

def parse_os(os_str: str) -> str:
    """Categorizes the Operating System."""
    if not isinstance(os_str, str):
        return 'Unknown'
        
    os_str_low = os_str.lower()
    if 'windows' in os_str_low:
        return 'Windows'
    if 'mac' in os_str_low or 'macos' in os_str_low:
        return 'macOS'
    if 'linux' in os_str_low or 'ubuntu' in os_str_low:
        return 'Linux'
    if 'chrome' in os_str_low:
        return 'ChromeOS'
    if 'dos' in os_str_low:
        return 'DOS/No OS'
    return 'Other'

def calculate_age(release_year: Optional[float]) -> Optional[int]:
    """Calculates laptop age."""
    if pd.isna(release_year):
        return None
    current_year = pd.Timestamp.now().year
    return current_year - int(release_year)

def create_heuristic_features(row: pd.Series) -> pd.Series:
    """Creates heuristic features like 'is_gaming' and 'is_ultrabook'."""
    # Gaming: Discrete GPU, 16GB+ RAM or high-end CPU
    row['is_gaming'] = 0
    if row['gpu_type'] == 'Discrete':
        if row['ram_gb'] >= 16 or row['cpu_series'] in ['Core i7', 'Core i9', 'Ryzen 7', 'Ryzen 9']:
            row['is_gaming'] = 1
        elif row['gpu_brand'] == 'NVIDIA' and 'RTX' in row['gpu_model']:
             row['is_gaming'] = 1

    # Ultrabook: Lightweight, SSD, small screen
    row['is_ultrabook'] = 0
    if (row['weight_kg'] <= 1.4 and 
        row['storage_type'] == 'SSD' and 
        row['display_size_in'] <= 14.0):
        row['is_ultrabook'] = 1
        
    return row

def calculate_cpu_score(row: pd.Series) -> int:
    """Creates a simple heuristic score for CPU power."""
    score = 0
    series_score = {
        'Core i9': 10, 'Ryzen 9': 10, 'M2 Max': 10, 'M1 Max': 10,
        'Core i7': 8, 'Ryzen 7': 8, 'M2 Pro': 9, 'M1 Pro': 9,
        'Core i5': 6, 'Ryzen 5': 6, 'M2': 7, 'M1': 7,
        'Core i3': 4, 'Ryzen 3': 4,
        'Pentium': 2, 'Celeron': 1, 'Athlon': 2,
    }
    score += series_score.get(row['cpu_series'], 0)
    
    if pd.notna(row['cpu_generation']):
        score += row['cpu_generation'] / 5.0 # Weight generation
        
    return int(score)