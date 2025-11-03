from google.cloud import bigquery
import pandas as pd
from pathlib import Path


def download_data():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    client = bigquery.Client(project='pablo-cdp-ppp')
    table_id = 'pablo-cdp-ppp.proyecto_final_cdp.restaurante'
    
    print("Downloading data from BigQuery...")
    query = f"SELECT * FROM `{table_id}`"
    df = client.query(query).to_dataframe()
    
    output_file = "data/raw/restaurante.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"Downloaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Saved to {output_file}")
    
    return df


def main():
    return download_data()


if __name__ == "__main__":
    main()