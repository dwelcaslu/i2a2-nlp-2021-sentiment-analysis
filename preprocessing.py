import matplotlib.pyplot as plt
import numpy as np


class SelectMajorCategories:
    def __init__(self, columns: list, perc: float = 0.1, minor_label='<other>', dropna=True):
        self.columns = columns if columns is not None else []
        self.perc = perc
        self.major_categories = {}
        self.minor_label = minor_label
        self.dropna = dropna

    def fit(self, x_df):
        for col in self.columns:
            col_value_counts = x_df[col].value_counts(dropna=self.dropna)
            col_major_counts = col_value_counts[col_value_counts > self.perc * x_df.shape[0]]
            self.major_categories[col] = col_major_counts.index
        return self

    def transform(self, x_df):
        x_df_ = x_df.copy()
        for col in self.columns:
            x_df_[col][~np.isin(x_df[col], self.major_categories[col])] = self.minor_label
        return x_df_


class CycleEncoder:
    def  __init__(self, period):
        self.period = period

    def fit(self, x_array):
        pass

    def transform(self, x_array):
        x_array_cos = np.cos(x_array * (2*np.pi/self.period))
        x_array_sin = np.sin(x_array * (2*np.pi/self.period))
        return x_array_cos, x_array_sin

