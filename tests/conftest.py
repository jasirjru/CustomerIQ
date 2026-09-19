"""Seal reserved data for the entire suite, including API worker threads."""
import pytest
from src.data.make_dataset import final_test_sealed


@pytest.fixture(autouse=True)
def seal_final_test():
    with final_test_sealed():
        yield
