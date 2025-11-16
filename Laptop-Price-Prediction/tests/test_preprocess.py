# File: tests/test_preprocess.py
"""
Tests for feature engineering functions in src/data/features.py
"""

import pytest
import pandas as pd
from src.data.features import (
    parse_ram, parse_storage, parse_cpu, parse_price, 
    parse_weight, parse_os, calculate_ppi
)

def test_parse_price():
    assert parse_price("₹74,990") == 74990.0
    assert parse_price("₹74,990.00") == 74990.0
    assert parse_price("$1,299") == 1299.0 * 83.0 # Using example rate
    assert parse_price("Free") is None
    assert parse_price(None) is None

def test_parse_ram():
    assert parse_ram("16 GB LPDDR5 RAM") == 16
    assert parse_ram("8GB") == 8
    assert parse_ram("RAM: 32 GB") == 32
    assert parse_ram("No RAM specified") is None
    assert parse_ram(None) is None

def test_parse_storage():
    assert parse_storage("512 GB SSD") == (512, "SSD")
    assert parse_storage("1 TB HDD") == (1024, "HDD")
    assert parse_storage("1 TB HDD + 256 GB SSD") == (1280, "Hybrid")
    assert parse_storage("256GB SSD") == (256, "SSD")
    assert parse_storage("2 TB") == (2048, "SSD") # Assumes SSD if not specified
    assert parse_storage("128 GB eMMC") == (128, "Other")
    assert parse_storage("None") == (None, "Unknown")
    
def test_parse_cpu():
    # Brand, Series, Generation
    assert parse_cpu("Intel Core i7 12th Gen") == ("Intel", "Core i7", 12)
    assert parse_cpu("Intel Core i5-1135G7") == ("Intel", "Core i5", 11)
    assert parse_cpu("AMD Ryzen 5 5600H") == ("AMD", "Ryzen 5", 5)
    assert parse_cpu("Apple M2 Pro") == ("Apple", "M2 Pro", 0)
    assert parse_cpu("Intel Celeron N4020") == ("Intel", "Celeron", None)
    assert parse_cpu("Qualcomm Snapdragon") == ("Other", "Unknown", None)
    
def test_parse_weight():
    assert parse_weight("1.86 kg") == 1.86
    assert parse_weight("2.1kg") == 2.1
    assert parse_weight("900 g") == 0.9
    assert parse_weight("1200grams") == 1.2
    assert parse_weight("Light") is None

def test_parse_os():
    assert parse_os("Windows 11 Home") == "Windows"
    assert parse_os("macOS Ventura") == "macOS"
    assert parse_os("Chrome OS") == "ChromeOS"
    assert parse_os("Ubuntu") == "Linux"
    assert parse_os("DOS") == "DOS/No OS"
    assert parse_os("PrimeOS") == "Other"

def test_calculate_ppi():
    # 15.6" 1920x1080
    assert calculate_ppi(15.6, "1920x1080") == 141.21
    # 13.3" 2560x1600 (MacBook Air)
    assert calculate_ppi(13.3, "2560x1600") == 227.0
    assert calculate_ppi(None, "1920x1080") is None
    assert calculate_ppi(14.0, "Full HD") is None