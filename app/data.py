import pandas as pd
from fastapi import UploadFile

class DataStore:
    def __init__(self):
        self.df = None

    def load_csv(self, file: UploadFile):
        self.df = pd.read_csv(file.file)
        return self.df

    def get(self):
        return self.df

    def metadata(self, df):
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        }
