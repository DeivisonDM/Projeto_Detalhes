import pandas as pd
import numpy as np
import os


def gerar_dados():

    # Verifica se o diretório de saída existe, caso contrário, cria
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Gera dados simulados
    np.random.seed(42)
    datas = pd.date_range(start="2023-01-01", periods=100, freq='W')
    vendas = np.random.poisson(lam=200, size=100) + 10 * np.sin(np.linspace(0, 10, 100))
    df = pd.DataFrame({'Data': datas, 'Vendas': vendas})
    df.set_index('Data', inplace=True)
    df.index.freq = 'W'

    df.to_csv("output/dados_vendas.csv")


gerar_dados()
