from google.cloud import bigquery

# Specify your project ID
client = bigquery.Client(project='pablo-cdp-ppp')

# Your table reference
table_id = 'pablo-cdp-ppp.proyecto_final_cdp.restaurante'

# Query or download your table
query = f"SELECT * FROM `{table_id}`"
df = client.query(query).to_dataframe()

query2 = f"SELECT * FROM `{table_id}` LIMIT 10"
df2= client.query(query2).to_dataframe()
print(df2.head())