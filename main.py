import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from gerar_dados import gerar_dados


df = pd.read_csv("output/dados_vendas.csv")

train = df.iloc[:-10]
test = df.iloc[-10:]

model = ARIMA(train['Vendas'], order=(2, 1, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)


mae = mean_absolute_error(test['Vendas'], forecast)
rmse = np.sqrt(mean_squared_error(test['Vendas'], forecast))

print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


plt.figure(figsize=(10, 4))
plt.plot(train.index, train['Vendas'], label='Treinamento')
plt.plot(test.index, test['Vendas'], label='Real')
plt.plot(test.index, forecast, label='Previsto', linestyle='--')
plt.title("Previsão de Vendas com ARIMA")
plt.xlabel("Data")
plt.ylabel("Unidades")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("output/previsao_vendas.png")
plt.show()

estoque_atual = 500
demanda_prevista = sum(forecast)

print("\nEstoque Atual:", estoque_atual)
print("Demanda Prevista (10 semanas):", round(demanda_prevista, 2))

if estoque_atual < demanda_prevista:
    print("⚠️ ALERTA: Estoque insuficiente para atender à demanda prevista!")
    alerta = "RUPTURA DE ESTOQUE"
else:
    print("✅ Estoque suficiente para o período previsto.")
    alerta = "ESTOQUE OK"


df_forecast = pd.DataFrame({
    'Data': test.index,
    'Demanda_Prevista': forecast,
    'Demanda_Real': test['Vendas'].values
})
df_forecast['Alerta'] = alerta
df_forecast.to_csv("output/previsao_demandas.csv", index=False)

log = {
    "Data_Execucao": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "Estoque_Atual": [estoque_atual],
    "Demanda_Prevista_Total": [round(demanda_prevista, 2)],
    "MAE": [round(mae, 2)],
    "RMSE": [round(rmse, 2)],
    "Status_Alerta": [alerta]
}
df_log = pd.DataFrame(log)
df_log.to_csv("output/log_execucao.csv", index=False)

print("\n✅ Resultados salvos na pasta 'output'.")
