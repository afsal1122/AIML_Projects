# src/recommend/recommender.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils import get_logger

logger = get_logger("RECOMMENDER")

PROCESSED_PATH = Path("data/processed/laptops_cleaned.csv")

class Recommender:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.feature_matrix = None
        self._load_data()
        self._prepare_features()
        
    def _load_data(self):
        """Load processed data"""
        try:
            if PROCESSED_PATH.exists():
                self.df = pd.read_csv(PROCESSED_PATH)
                # Clean the data
                self.df = self.df.dropna(subset=['Price', 'RAM_GB', 'Storage_capacity_GB'])
                logger.info(f"Loaded data: {len(self.df)} laptops")
            else:
                logger.error("Processed data not found")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            
    def _prepare_features(self):
        """Prepare feature matrix for recommendations"""
        if self.df is None or self.df.empty:
            return
            
        try:
            # Select relevant features for recommendation
            feature_columns = [
                'Processor_gen_num', 'RAM_GB', 'Storage_capacity_GB', 
                'Graphics_GB', 'Display_size_inches', 'ppi', 'Price'
            ]
            
            # Use only columns that exist and have data
            available_columns = [col for col in feature_columns if col in self.df.columns and self.df[col].notna().any()]
            
            # Fill missing values with median
            features_df = self.df[available_columns].copy()
            for col in available_columns:
                if features_df[col].isna().any():
                    features_df[col] = features_df[col].fillna(features_df[col].median())
            
            # Scale features
            self.feature_matrix = self.scaler.fit_transform(features_df)
            self.feature_columns = available_columns
            logger.info(f"Prepared features: {len(available_columns)} columns")
            
        except Exception as e:
            logger.error(f"Failed to prepare features: {e}")

    def _calculate_usage_score(self, laptop, usage_type):
        """Calculate score based on usage type with intelligent weighting"""
        score = 0
        
        if usage_type == "gaming":
            # Gaming: Emphasize GPU, Processor, RAM
            gpu_score = laptop.get('Graphics_GB', 0) * 3.0
            processor_score = laptop.get('Processor_gen_num', 0) * 2.0
            ram_score = laptop.get('RAM_GB', 0) * 1.5
            score = gpu_score + processor_score + ram_score
            
        elif usage_type == "programming":
            # Programming: Emphasize RAM, Processor, Storage
            ram_score = laptop.get('RAM_GB', 0) * 3.0
            processor_score = laptop.get('Processor_gen_num', 0) * 2.0
            storage_score = laptop.get('Storage_capacity_GB', 0) * 0.5
            score = ram_score + processor_score + storage_score
            
        elif usage_type == "content-creation":
            # Content Creation: Balanced high specs
            processor_score = laptop.get('Processor_gen_num', 0) * 2.0
            ram_score = laptop.get('RAM_GB', 0) * 2.0
            storage_score = laptop.get('Storage_capacity_GB', 0) * 1.5
            gpu_score = laptop.get('Graphics_GB', 0) * 1.5
            score = processor_score + ram_score + storage_score + gpu_score
            
        elif usage_type == "lightweight":
            # Lightweight: Emphasize portability (smaller display, newer processor)
            display_score = (1 / laptop.get('Display_size_inches', 15)) * 3.0  # Prefer smaller displays
            processor_score = laptop.get('Processor_gen_num', 0) * 2.0
            weight_penalty = laptop.get('Display_size_inches', 15) * 0.5  # Penalize larger displays
            score = display_score + processor_score - weight_penalty
            
        else:  # general
            # General use: Balanced approach with value for money
            processor_score = laptop.get('Processor_gen_num', 0) * 1.5
            ram_score = laptop.get('RAM_GB', 0) * 1.5
            storage_score = laptop.get('Storage_capacity_GB', 0) * 1.0
            value_score = (1 / (laptop['Price'] / 100000)) * 2.0  # Value for money
            score = processor_score + ram_score + storage_score + value_score
            
        return score

    def _apply_must_haves(self, filtered_df, must_haves):
        """Apply must-have filters intelligently"""
        if "SSD" in must_haves:
            filtered_df = filtered_df[filtered_df['Storage_type'].str.contains('SSD', case=False, na=False)]
        
        if "Dedicated GPU" in must_haves:
            # Consider GPUs with >2GB as dedicated, or specific brands
            gpu_condition = (
                (filtered_df['Graphics_GB'] > 2) |
                (filtered_df['Graphics_brand'].isin(['NVIDIA', 'AMD'])) |
                (filtered_df['Graphics_name'].str.contains('RTX|GTX|Radeon', case=False, na=False))
            )
            filtered_df = filtered_df[gpu_condition]
        
        if "High RAM (16GB+)" in must_haves:
            filtered_df = filtered_df[filtered_df['RAM_GB'] >= 16]
        
        if "Latest Processor" in must_haves:
            # Consider processors from last 2 generations as latest
            filtered_df = filtered_df[filtered_df['Processor_gen_num'] >= 12]
            
        return filtered_df

    def _ensure_brand_diversity(self, recommendations, max_per_brand=2):
        """Ensure we don't get too many products from the same brand - only for 'All Brands'"""
        if recommendations.empty:
            return recommendations
            
        diversified = []
        brand_counts = {}
        
        for _, laptop in recommendations.iterrows():
            brand = laptop['Brand']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            
            if brand_counts[brand] <= max_per_brand:
                diversified.append(laptop)
            else:
                # Skip if we already have enough from this brand
                continue
                
        return pd.DataFrame(diversified)

    def _ensure_price_distribution(self, recommendations, min_price, max_price, target_count):
        """Ensure recommendations are spread across the price range"""
        if recommendations.empty or len(recommendations) <= target_count:
            return recommendations
            
        # Create price buckets
        price_range = max_price - min_price
        if price_range > 0:
            bucket_size = price_range / 3  # 3 price segments: low, medium, high
        else:
            bucket_size = 1
            
        buckets = {
            'low': (min_price, min_price + bucket_size),
            'medium': (min_price + bucket_size, min_price + 2 * bucket_size),
            'high': (min_price + 2 * bucket_size, max_price)
        }
        
        selected = []
        items_per_bucket = max(1, target_count // 3)
        
        for bucket_name, (bucket_min, bucket_max) in buckets.items():
            bucket_items = recommendations[
                (recommendations['Price'] >= bucket_min) & 
                (recommendations['Price'] <= bucket_max)
            ].head(items_per_bucket)
            
            selected.extend(bucket_items.to_dict('records'))
        
        # If we don't have enough, add the highest scored ones
        if len(selected) < target_count:
            remaining = target_count - len(selected)
            additional = recommendations.head(remaining)
            selected.extend(additional.to_dict('records'))
            
        return pd.DataFrame(selected)

    def recommend_with_brand(self, min_price, max_price, usage_types, must_haves, brand_filter=None, top_k=8):
        """Generate recommendations with optional brand filtering"""
        if self.df is None:
            logger.error("No data available")
            return pd.DataFrame()
            
        # Filter by price first
        filtered = self.df[
            (self.df['Price'] >= min_price) & 
            (self.df['Price'] <= max_price)
        ].copy()
        
        if filtered.empty:
            logger.warning("No laptops in price range")
            return pd.DataFrame()
        
        # Apply brand filter if specified
        if brand_filter and brand_filter != "All Brands":
            filtered = filtered[filtered['Brand'] == brand_filter]
            if filtered.empty:
                logger.warning(f"No {brand_filter} laptops in price range")
                return pd.DataFrame()
        
        # Apply must-have filters
        filtered = self._apply_must_haves(filtered, must_haves)
        
        if filtered.empty:
            logger.warning("No laptops match must-have criteria")
            return pd.DataFrame()
        
        usage_type = usage_types[0] if usage_types else "general"
        
        # Calculate scores for each laptop
        scores = []
        for _, laptop in filtered.iterrows():
            base_score = self._calculate_usage_score(laptop, usage_type)
            
            # Adjust score based on value for money
            price_range = max_price - min_price
            if price_range > 0:
                price_ratio = (laptop['Price'] - min_price) / price_range
                value_bonus = (1 - price_ratio) * 3.0
            else:
                value_bonus = 0
            
            final_score = base_score + value_bonus
            scores.append(final_score)
        
        filtered['Score'] = scores
        
        # Apply diversity only if no specific brand is selected (All Brands)
        if not brand_filter or brand_filter == "All Brands":
            # Get initial top recommendations (more than needed for diversity)
            initial_recommendations = filtered.nlargest(top_k * 3, 'Score')
            
            # Ensure brand diversity
            diverse_recommendations = self._ensure_brand_diversity(initial_recommendations, max_per_brand=2)
            
            # Ensure price distribution
            final_recommendations = self._ensure_price_distribution(
                diverse_recommendations, min_price, max_price, top_k
            )
        else:
            # For specific brand, take as many as requested (no brand limit)
            final_recommendations = filtered.nlargest(top_k, 'Score')
        
        # Normalize scores to 0-1 for better display
        if not final_recommendations.empty:
            max_score = final_recommendations['Score'].max()
            min_score = final_recommendations['Score'].min()
            if max_score > min_score:
                final_recommendations['Score'] = (final_recommendations['Score'] - min_score) / (max_score - min_score)
            else:
                final_recommendations['Score'] = 1.0
        
        brand_info = f"for {brand_filter}" if brand_filter and brand_filter != "All Brands" else "across brands"
        logger.info(f"Generated {len(final_recommendations)} recommendations {brand_info} for {usage_type}")
        
        # Select and return relevant columns
        result_columns = ['Brand', 'Price', 'Processor_name', 'RAM_GB', 
                         'Storage_capacity_GB', 'Graphics_name', 'Score']
        
        available_columns = [col for col in result_columns if col in final_recommendations.columns]
        return final_recommendations[available_columns].reset_index(drop=True)

    def recommend(self, min_price, max_price, usage_types, must_haves, top_k=8):
        """Main recommendation method - uses brand diversity by default"""
        return self.recommend_with_brand(min_price, max_price, usage_types, must_haves, None, top_k)

    def get_available_brands(self):
        """Get list of available brands in the dataset"""
        if self.df is not None and 'Brand' in self.df.columns:
            return sorted(self.df['Brand'].unique().tolist())
        return []

    def get_brand_price_range(self, brand):
        """Get price range for a specific brand"""
        if self.df is not None and brand in self.df['Brand'].values:
            brand_data = self.df[self.df['Brand'] == brand]
            return {
                'min_price': int(brand_data['Price'].min()),
                'max_price': int(brand_data['Price'].max()),
                'avg_price': int(brand_data['Price'].mean()),
                'count': len(brand_data)
            }
        return None