from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        
        self.columns_to_drop = [
            'id_persona',
            'nombre', 
            'apellido',
            'telefono_contacto',
            'correo_electronico'
        ]
        
        self.numerical_cols = [
            'edad',
            'frecuencia_visita',
            'promedio_gasto_comida',
            'ingresos_mensuales'
        ]
        
        self.ordinal_cols = ['estrato_socioeconomico']
        
        self.nominal_cols = [
            'genero',
            'ciudad_residencia',
            'ocio',
            'consume_licor',
            'preferencias_alimenticias',
            'membresia_premium',
            'tipo_de_pago_mas_usado'
        ]
    
    def load_data(self):
        self.df = pd.read_parquet(self.input_path)
        print(f"Loaded {len(self.df)} rows")
        return self
    
    def remove_unnecessary_columns(self):
        self.df = self.df.drop(columns=self.columns_to_drop)
        print(f"Dropped {len(self.columns_to_drop)} columns")
        return self
    
    def impute_missing_values(self):
        num_imputer = SimpleImputer(strategy='median')
        self.df[self.numerical_cols] = num_imputer.fit_transform(
            self.df[self.numerical_cols]
        )
        
        all_categorical = self.ordinal_cols + self.nominal_cols
        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.df[all_categorical] = cat_imputer.fit_transform(
            self.df[all_categorical]
        )
        
        print(f"Imputed missing values")
        return self
    
    def encode_ordinal_variables(self):
        for col in self.ordinal_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
        
        print(f"Encoded {len(self.ordinal_cols)} ordinal variables")
        return self
    
    def encode_nominal_variables(self):
        self.df = pd.get_dummies(
            self.df, 
            columns=self.nominal_cols,
            drop_first=True,
            dtype=int
        )
        
        print(f"One-hot encoded {len(self.nominal_cols)} nominal variables")
        return self
    
    def validate_data(self):
        missing = self.df.isnull().sum().sum()
        print(f"Remaining missing values: {missing}")
        print(f"Final shape: {self.df.shape}")
        print(f"All columns are numeric: {all(self.df.dtypes.apply(lambda x: np.issubdtype(x, np.number)))}")
        return self
    
    def save_data(self):
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(self.output_path, index=False)
        print(f"Saved to {self.output_path}")
        return self
    
    def run_pipeline(self):
        self.load_data()
        self.remove_unnecessary_columns()
        self.impute_missing_values()
        self.encode_ordinal_variables()
        self.encode_nominal_variables()
        self.validate_data()
        self.save_data()
        return self.df


def main():
    preprocessor = DataPreprocessor(
        input_path='data/raw/restaurante.parquet',
        output_path='data/processed/restaurante_clean.parquet'
    )
    
    df_clean = preprocessor.run_pipeline()
    return df_clean


if __name__ == "__main__":
    main()