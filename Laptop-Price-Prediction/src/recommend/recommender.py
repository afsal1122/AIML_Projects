import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from src.models.persistence import load_model

DATA_PATH = Path("data/processed/laptops_cleaned.csv")
PIPELINE_PATH = Path("models/preprocessing_pipeline.pkl")

class Recommender:
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
        self.pipeline = load_model(PIPELINE_PATH) if PIPELINE_PATH.exists() else None
        self.df_feats = None
        if self.df is not None and self.pipeline is not None:
            self._prep_features()
            
    def _prep_features(self):
        # We only transform, we don't refit
        feats = self.pipeline.get_feature_names_out()
        data_proc = self.pipeline.transform(self.df)
        
        scaler = MinMaxScaler()
        self.df_feats = pd.DataFrame(scaler.fit_transform(data_proc), columns=feats, index=self.df.index)
        
        # Add heuristics back for easier scoring
        for col in ['is_gaming', 'ram_gb', 'weight_kg', 'is_ultrabook']:
            if col in self.df.columns:
                self.df_feats[col] = self.df[col]

    def recommend(self, min_p, max_p, use_cases, must_haves, top_k=5):
        if self.df is None: return pd.DataFrame()
        
        # 1. Hard Filter
        mask = (self.df['price'] >= min_p) & (self.df['price'] <= max_p)
        
        if 'Dedicated GPU' in must_haves:
            mask &= (self.df['gpu_type'] == 'Discrete')
        if 'SSD' in must_haves:
            mask &= (self.df['storage_type'].str.contains('SSD', case=False, na=False))
            
        candidates = self.df[mask].copy()
        if candidates.empty: return pd.DataFrame()
        
        # 2. Scoring
        scores = np.zeros(len(candidates))
        cand_feats = self.df_feats.loc[candidates.index]
        
        if 'gaming' in use_cases:
            scores += cand_feats.get('is_gaming', 0) * 3.0
            scores += cand_feats.get('ram_gb', 0) * 0.1
            scores += cand_feats.get('cpu_score', 0) * 1.0
            
        if 'programming' in use_cases:
            scores += cand_feats.get('ram_gb', 0) * 0.2
            scores += cand_feats.get('cpu_score', 0) * 1.0
            
        if 'lightweight' in use_cases:
            scores += cand_feats.get('is_ultrabook', 0) * 3.0
            # Inverse weight (lighter is better)
            scores += (1 / (cand_feats.get('weight_kg', 1) + 0.1)) * 2.0
            
        # Rating boost
        scores += (candidates['user_rating'].fillna(3.0) / 5.0)
        
        candidates['Score'] = scores
        recs = candidates.sort_values('Score', ascending=False).head(top_k)
        
        # Return clean view
        recs['Rank'] = range(1, len(recs)+1)
        return recs[['Rank', 'brand', 'model', 'price', 'ram_gb', 'cpu_series', 'url', 'Score']]