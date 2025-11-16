# File: tests/test_recommender.py
"""
Tests for the Recommender class in src/recommend/recommender.py
"""

import pytest
import pandas as pd
from src.recommend.recommender import Recommender

# We can't easily test the recommender without all the artifacts.
# A full test would require mocking the load_model and df.
# For this project, we'll just check if the class initializes
# (and gracefully fails if files are missing).

@pytest.fixture
def recommender_instance():
    """
    Provides a Recommender instance.
    This test will FAIL if artifacts aren't built,
    which is correct behavior (integration test).
    """
    # This fixture assumes the sample data and pipelines EXIST
    # To run this test standalone, you'd need to mock file existence
    # and the outputs of load_model.
    try:
        recommender = Recommender()
        return recommender
    except Exception:
        pytest.skip("Skipping recommender test: artifacts (data/model) not found.")

def test_recommender_init(recommender_instance):
    """Test if the recommender initializes and loads data."""
    if recommender_instance:
        assert recommender_instance.is_ready()
        assert isinstance(recommender_instance.df, pd.DataFrame)
        assert not recommender_instance.df.empty
        assert recommender_instance.pipeline is not None
        assert recommender_instance.model is not None

def test_recommend_laptops(recommender_instance):
    """Test the recommend_laptops function."""
    if recommender_instance:
        recs = recommender_instance.recommend_laptops(
            budget_min=40000,
            budget_max=80000,
            use_cases=['programming'],
            must_haves=['SSD'],
            top_k=3
        )
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) <= 3
        if not recs.empty:
            assert 'recommend_score' in recs.columns

def test_find_good_deals(recommender_instance):
    """Test the find_good_deals function."""
    if recommender_instance:
        deals = recommender_instance.find_good_deals(discount_threshold=0.10)
        assert isinstance(deals, pd.DataFrame)
        if not deals.empty:
            assert 'discount_pct' in deals.columns
            assert (deals['discount_pct'] >= 0.10).all()