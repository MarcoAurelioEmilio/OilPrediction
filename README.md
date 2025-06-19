# OilPredictions
Este projeto busca fornecer insights relevantes com base em previsões estatísticas avançadas e conjuntos de dados abrangentes a respeito da demanda das refinarias americanas por petróleo, assim como, gasolina e diesel/heating oil nos EUA.

---
# Apresentando as pastas

## **📂data**

Esta pasta está dividida em duas pastas **📂raw** e **📂processed**.

A pasta **📂raw** contém os dados brutos, a base de dados é muito grande para ser upada no GitHub, acesse em: 
[Joint Organizations Data Initiative - JODI](https://www.jodidata.org/oil/database/data-downloads.aspx) e faça o Download de **"extended csv - primary"** e **"extended csv - secondary**.

A pasta **📂processed** contém os dados já processados e tratados. Sendo **Processed_JODI.csv** a base de dados tratada e **jodi_dataset_predictions.csv** a base de dados tratada e com a adição dos dados previstos com base nos modelos preditivos que foram usados.

---
## 📂**src** / 📂**aux_functions_production**

Esta pasta contém os scripts **import_data.ipynb** e **model.ipynb**
**import_data.ipynb** é o script em python onde eu realizei o tratamento dos dados **raw** da JODI.
**model.ipynb** é onde contém todo o modelo

---
## 📂power bi

Esta pasta contém a **Captura de tela pbi.png** para vizualizar o dashboard e também contém o próprio **dashboard jodi.pbix** onde é possível interagir com a demanda de petróleo, gasolina e diesel/heating oil por meio de um gráfico de linhas e com o recorte de tempo desejado. 

O filtro de tempo tem dados que vão de janeiro de 2002 até setembro de 2025. Sendo os dados a partir de março de 2025, dados previstos com base em todo o modelo estatístico desenvolvido para esse projeto. 

---

# Explicando o projeto

---

## 🧪 Bibliotecas Utilizadas

Este projeto utiliza uma série de bibliotecas Python voltadas à análise de dados, modelagem estatística e visualização gráfica:

### 📦 Principais Bibliotecas

| Biblioteca | Descrição |
|------------|-----------|
| **pandas** | Manipulação e análise de dados tabulares (DataFrames). |
| **numpy** | Operações numéricas e criação de arrays/matrizes. |
| **statsmodels** | Modelos estatísticos e testes como ARIMA, SARIMA, ADF, Durbin-Watson. |
| **matplotlib.pyplot** | Criação de gráficos estáticos, incluindo linhas, barras e datas. |
| **seaborn** | Visualizações estatísticas com gráficos mais elaborados baseados em Matplotlib. |
| **scipy.stats** | Testes estatísticos como o de Shapiro-Wilk para normalidade. |
| **os** | Acesso e manipulação de diretórios e arquivos do sistema. |
| **mpl_toolkits.mplot3d** | Criação de gráficos tridimensionais com Matplotlib. |
| **pmdarima** | Modelagem automática de séries temporais (auto_arima). |

---

### 🔍 Importações Específicas

| Comando Importado | Origem | Utilidade |
|-------------------|--------|----------|
| `shapiro` | `scipy.stats` | Teste de normalidade dos resíduos (Shapiro-Wilk). |
| `durbin_watson` | `statsmodels.stats.stattools` | Verifica autocorrelação dos resíduos. |
| `adfuller` | `statsmodels.tsa.stattools` | Teste de estacionariedade (ADF - Dickey-Fuller aumentado). |
| `auto_arima` | `pmdarima` | Escolha automática dos melhores parâmetros para o modelo ARIMA/SARIMA. |
| `ARIMA` | `statsmodels.tsa.arima.model` | Estimação manual de modelos ARIMA. |
| `Axes3D` | `mpl_toolkits.mplot3d` | Geração de gráficos 3D. |
| `mdates` | `matplotlib.dates` | Manipulação de datas em eixos de tempo nos gráficos. |

---
## Regressão Linear

```math
CRUDEOIL\_REFINOBS = \beta_0 + \beta_1 \times GASOLINE\_TOTDEMO + \beta_2 \times GASDIES\_TOTDEMO + \varepsilon
```
### Regressão Linear para Demanda de Refinarias.
Neste modelo, utilizei uma regressão tendo como variável dependente a CRUDEOIL_REFINOBS, 
que representa a demanda por petróleo das refinarias americanas. As variáveis explicativas serão a demanda total de gasolina e a demanda total de diesel, 
escolhidas por representarem aproximadamente 80% do volume processado em uma refinaria.

### Confira em **model.ipynb** para observar os gráficos, os cenários hipotéticos e as interpretações.

--- 
## Diesel - Arima

```math
(1 - \phi_1 L - \phi_2 L^2 - \ldots - \phi_p L^p) \nabla^d y_t = (1 + \theta_1 L + \theta_2 L^2 + \ldots + \theta_q L^q) \epsilon_t
```

Utilizei o modelo ARIMA para séries temporais
### Confira em **model.ipynb** para observar os gráficos, interpretações e previsões.

---
## Gasolina - Sarima

```math
\Phi_P(L^s) \cdot \phi_p(L) \cdot \nabla^d \nabla_s^D y_t = \Theta_Q(L^s) \cdot \theta_q(L) \cdot \epsilon_t
```

Utilizei o modelo SARIMA para sérires temporais, levando em consideração que há muita sazonalidade na demanda por gasolina.
### Confira em **model.ipynb** para observar os gráficos, interpretações e previsões.

---
# Conferindo o Power B.I 

Após eu gerar o modelo e salvar as previsões em:
├── 📂data/
│ └── 📂processed/ **jodi_dataset_predictions.csv**

Realizei com essa base de dados o 📊B.I da demanda de petróleo, gasolina e diesel/heating oil nos EUA.

Você pode conferir o Arquivo em:
├── 📂power bi/ **dashboard jodi.pbix**

# ✅ Considerações Finais

A aplicação dos modelos SARIMA e ARIMA aos dados de consumo de combustíveis revelou-se uma abordagem eficaz para capturar a sazonalidade e as tendências das séries temporais analisadas. Os resultados indicam que há padrões regulares ao longo do tempo, permitindo previsões confiáveis de curto prazo.

A modelagem permitiu:

- Identificar os períodos de maior e menor consumo.
- Antecipar possíveis variações sazonais que impactam o mercado de combustíveis.
- Gerar previsões com intervalos de confiança que auxiliam na tomada de decisão.

Com mais tempo, este projeto pode ser expandido com:

- Inclusão de variáveis externas.
- Avaliação de modelos multivariados (VAR, SARIMAX com exógenas).
- Atualização contínua com dados mais recentes da base JODI ou de outras fontes.

Em suma, o modelo desenvolvido oferece uma base sólida para análises preditivas no setor, com potencial de apoio à formulação de estratégias de mercado.






