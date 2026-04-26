import os
import yaml
import pandas as pd
from datasets import load_dataset
import logging
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class to manage the acquisition and local storage of 
    research datasets.
    """
    def __init__(self, config):
        self.config = config
        self.dataset_name = self.config["dataset"]["path"]
        self.version = self.config["dataset"]["version"]
        self.subset_size = self.config["dataset"]["subset_size"]
        self.raw_path = os.path.join("data", "raw", "raw_samples.csv")
    
    def fetch_from_cloud(self):
        """Downloads the dataset from Hugging Face."""
        print(f"--- Fetching {self.dataset_name} ---")
        dataset = load_dataset(
            self.dataset_name,
            self.version,
            split=self.config["dataset"]["split"]
        )
        df = pd.DataFrame(dataset[:self.subset_size])
        return df
    
    def get_data(self):
        """
        Returns local data if exists, otherwise fetches new data.
        """
        if os.path.exists(self.raw_path):
            print(f"Loading data from local cache: {self.raw_path}")
            return pd.read_csv(self.raw_path)
        
        df = self.fetch_from_cloud()
        self._save_locally(df)
        return df
    
    def _save_locally(self, df):
        """Private method to cache data."""
        os.makedirs(os.path.dirname(self.raw_path), exist_ok=True)
        df.to_csv(self.raw_path, index=False)
        print(f"Data cached at {self.raw_path}")
    
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    loader = DataLoader(config=config)
    data = loader.get_data()
    print(data.head())