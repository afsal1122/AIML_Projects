# src/data/features.py
import pandas as pd
import re
import numpy as np

def clean_processor_gen(val):
    """
    Robustly parses 'Processor_gen' into consistent numeric scale.
    Apple M-series: M1 -> 101, M2 -> 102, etc.
    Intel/AMD: Extract generation number
    """
    if pd.isna(val): 
        return np.nan
        
    val_str = str(val).strip()
    
    # Apple Silicon Mapping (100+ scheme)
    apple_match = re.search(r'M\s*(\d+)', val_str, re.IGNORECASE)
    if apple_match:
        try:
            return 100 + int(apple_match.group(1))  # M1=101, M2=102, etc.
        except:
            pass
    
    # Intel Special mappings
    if 'Meteor Lake' in val_str or 'Ultra' in val_str:
        return 14
    
    # Standard generation patterns
    match = re.search(r'(\d+)(?:th|nd|rd|st)?\s*Gen', val_str, re.IGNORECASE)
    if match: 
        return int(match.group(1))
    
    match_gen_x = re.search(r'Gen\s*(\d+)', val_str, re.IGNORECASE)
    if match_gen_x: 
        return int(match_gen_x.group(1))

    # Try to extract any digits
    digits = re.findall(r'\d+', val_str)
    if digits:
        try:
            return int(digits[0])
        except:
            pass

    return np.nan

def clean_memory(val):
    """Extracts numeric RAM value (e.g., '16GB' -> 16)."""
    if pd.isna(val): 
        return np.nan
    nums = re.findall(r'\d+', str(val))
    return int(nums[0]) if nums else np.nan

def clean_storage(val):
    """Extracts storage in GB."""
    if pd.isna(val): 
        return np.nan
    val_str = str(val).lower()
    nums = re.findall(r'\d+', val_str)
    if not nums: 
        return np.nan
    amount = int(nums[0])
    if 'tb' in val_str: 
        amount *= 1024
    return amount

def clean_display(val):
    """Extracts display size."""
    if pd.isna(val): 
        return np.nan
    try:
        return float(val)
    except:
        nums = re.findall(r'\d+\.?\d*', str(val))
        return float(nums[0]) if nums else np.nan

def extract_vram(gpu_name):
    """Extract VRAM from GPU name."""
    if pd.isna(gpu_name):
        return np.nan
    try:
        match = re.search(r'(\d+)\s*GB', str(gpu_name), re.IGNORECASE)
        if match:
            return int(match.group(1))
    except:
        pass
    return np.nan