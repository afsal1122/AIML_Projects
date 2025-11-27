import pandas as pd
import sys

try:
    df = pd.read_csv('data/raw/training_dataset.csv')
    print("Columns:", df.columns.tolist())
    
    # Check for Core_per_processor
    # Note: The preprocess script handles case sensitivity and stripping, so we should check raw names
    print("\nRaw Columns:", df.columns.tolist())
    
    # Check for potential matches for Core_per_processor
    print("\nChecking Core_per_processor:")
    found_core = False
    for col in df.columns:
        if 'core' in col.lower():
            print(f"Found column '{col}': non-null count = {df[col].count()}")
            print(f"Sample: {df[col].head().tolist()}")
            found_core = True
    if not found_core:
        print("No column containing 'core' found.")

    # Check for Graphics_GB
    print("\nChecking Graphics_GB:")
    if 'Graphics_GB' in df.columns:
        print("Graphics_GB non-null:", df['Graphics_GB'].count())
    else:
        print("Graphics_GB not in columns")
        
    # Check Graphics_name for extraction
    print("\nChecking Graphics_name:")
    found_graphics = False
    for col in df.columns:
        if 'graphics' in col.lower() and 'name' in col.lower():
            print(f"Found column '{col}':")
            print(df[col].head(10).tolist())
            found_graphics = True
    if not found_graphics:
        print("No column containing 'graphics' and 'name' found.")
        
except Exception as e:
    print(f"Error: {e}")
