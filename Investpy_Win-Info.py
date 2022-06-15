import pandas as pd
import investpy as inv

print('Iniciando extração de dados:')

indices=inv.search_quotes(text='WIN',products=['indices'],countries=['brazil'],n_results=50)

for indice in indices[0:1]:
    print('===================================================')

WIN=indice.retrieve_historical_data(from_date='01/01/1900',to_date='20/04/2024')

WIN.to_csv('WIN.csv')

print('Dados extraidos com sucesso!')
