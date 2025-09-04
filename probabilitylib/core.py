import pandas as pd
import numpy as np

class ProbabilitySpace():
   
    def __init__(self, df, normalise=True):
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
    
        self.df = df.copy()
        self.variables = {}
    
        if 'weights' not in self.df.columns:
            self.df['weights'] = 1.0 / len(self.df)
    
        self.df = self.df.dropna(subset=['weights'])
    
        self.n = len(self.df)
    
        if normalise and not np.isclose(self.df["weights"].sum(), 1.0):
            print("weights column does not sum to 1. Normalisation procedure enabled")
            self.df['weights'] = self.df["weights"] / self.df["weights"].sum()
    

    def size(self):
        return self.df.shape[0]

    def columns(self):
        return list(self.df.columns)

    def head(self, n=5):
        return self.df.head(n)

    def describe(self):
        return self.df.describe()

    def define_random_variable(self, name: str, func):        
        try:
            result = func(self.df)
            if not isinstance(result, pd.Series):
                raise ValueError("The function must return a pandas Series.")
            if len(result) != self.n:
                raise ValueError("Length of resulting Series must match DataFrame rows.")
            self.variables[name] = result
            self.df[name] = result
        except Exception as e:
            raise ValueError(f"Failed to define variable '{name}': {str(e)}")

    def get_variable(self, name: str) -> pd.Series:
        if name in self.df.columns:
            return self.df[name]
        else:
            raise KeyError(f"Variable '{name}' not found.")
        
   