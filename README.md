# Detecção de Fraudes em Cartões de Crédito

Projeto de Machine Learning para identificar transações fraudulentas com a base
[Credit Card Fraud Detection](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv).

**Tecnologias:** Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, imbalanced-learn, Matplotlib e Seaborn.

## O problema

A base tem 284.807 transações, e apenas 492 (0,17%) são fraudes. Um modelo que
responde "não é fraude" para tudo teria cerca de 99,8% de acurácia e encontraria
zero fraudes. Por isso a acurácia foi descartada como métrica principal e a
avaliação foca na classe de fraude:

- **Recall:** das fraudes reais, quantas o modelo encontrou.
- **Precisão:** das transações marcadas como fraude, quantas realmente eram.
- **F1:** equilíbrio entre as duas.
- **PR-AUC:** área sob a curva precisão x recall, a mais informativa em dados desbalanceados.

Exemplo do problema: a primeira Regressão Logística (com `class_weight='balanced'`)
atingiu 97,5% de acurácia e recall de 0,92, mas precisão de apenas 0,06. Para cada
fraude real, cerca de 15 transações normais eram bloqueadas.

## Preparação dos dados

- `Amount` transformado com `np.log1p` para reduzir a assimetria.
- Variáveis padronizadas com `StandardScaler`. [ ] Confirme se o scaler ficou dentro do Pipeline
  (ajustado só no treino) ou se foi aplicado antes da divisão, e descreva o que fez de fato.
- Divisão 80/20 com `stratify`: treino com 227.845 transações (394 fraudes) e teste com
  56.962 transações (98 fraudes).
- Desbalanceamento tratado com `class_weight='balanced'` (Random Forest e Regressão Logística)
  e `scale_pos_weight` (XGBoost).

## Comparação dos modelos (classe de fraude, base de teste, limiar 0,5)

| Modelo | Precisão | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Regressão Logística (com `class_weight='balanced'`) | 0,0603 | 0,9184 | 0,1131 | 0,9711 | - |
| Regressão Logística (sem peso) [ ] confirmar | 0,8095 | 0,6939 | 0,7473 | 0,9493 | 0,7257 |
| Random Forest | 0,9610 | 0,7551 | 0,8457 | 0,9576 | 0,8555 |
| XGBoost | 0,8723 | 0,8367 | 0,8542 | 0,9775 | 0,8760 |
| Random Forest ajustado (GridSearchCV) | 0,76 | 0,86 | 0,80 | - | 0,8041 |

**Leitura:**
- **XGBoost** teve o melhor equilíbrio no limiar padrão: encontrou 82 das 98 fraudes, com 12 falsos alarmes,
  e lidera F1, ROC-AUC e PR-AUC.
- **Random Forest** foi o mais preciso: 74 fraudes encontradas e só 3 falsos alarmes, mas perdeu 24.
- **Regressão Logística com peso** achou 90 fraudes, mas gerou 1.403 falsos alarmes. Seu ROC-AUC (0,97) é
  quase igual ao do XGBoost (0,98) apesar do F1 de 0,11, o que mostra por que o ROC-AUC engana em dados desbalanceados.
- **GridSearchCV** (escolhendo por recall) subiu o recall do Random Forest de 0,76 para 0,86, mas a precisão caiu
  de 0,96 para 0,76 e a PR-AUC de 0,86 para 0,80. O melhor `max_depth` foi 10, o menor valor da grade.
  A busca rodou numa amostra do treino (todas as fraudes + 40 mil normais) por limitação de tempo no Colab,
  e o modelo final foi retreinado com todo o treino.
- O teste tem só 98 fraudes: cada fraude vale cerca de 1 ponto percentual de recall, então diferenças de
  poucas fraudes entre modelos não são conclusivas.

## Limiar de decisão

O limiar mudou o resultado de forma diferente em cada modelo:

**XGBoost**

| Limiar | Precisão | Recall | F1 | Fraudes | Falsos alarmes |
|---|---|---|---|---|---|
| 0,1 | 0,7778 | 0,8571 | 0,8155 | 84 | 24 |
| 0,2 | 0,8469 | 0,8469 | 0,8469 | 83 | 15 |
| 0,3 | 0,8557 | 0,8469 | 0,8513 | 83 | 14 |
| 0,5 | 0,8723 | 0,8367 | 0,8542 | 82 | 12 |
| 0,7 | 0,8817 | 0,8367 | 0,8586 | 82 | 11 |

**Random Forest em Pipeline**

| Limiar | Precisão | Recall | F1 | Fraudes | Falsos alarmes |
|---|---|---|---|---|---|
| 0,1 | 0,7478 | 0,8776 | 0,8075 | 86 | 29 |
| 0,2 | 0,8557 | 0,8469 | 0,8513 | 83 | 14 |
| 0,3 | 0,9213 | 0,8367 | 0,8770 | 82 | 7 |
| 0,4 | 0,9512 | 0,7959 | 0,8667 | 78 | 4 |
| 0,5 | 0,9610 | 0,7551 | 0,8457 | 74 | 3 |

**Regressão Logística com `class_weight='balanced'`**

| Limiar | Precisão | Recall | F1 | Fraudes | Falsos alarmes |
|---|---|---|---|---|---|
| 0,1 | 0,0080 | 0,9490 | 0,0158 | 93 | 11.553 |
| 0,3 | 0,0262 | 0,9184 | 0,0510 | 90 | 3.342 |
| 0,5 | 0,0603 | 0,9184 | 0,1131 | 90 | 1.403 |
| 0,7 | 0,1208 | 0,9082 | 0,2132 | 89 | 648 |
| 0,9 | 0,2486 | 0,8878 | 0,3884 | 87 | 263 |

- **Random Forest:** baixar o limiar de 0,5 para 0,3 elevou o recall de 0,7551 para 0,8367 (de 74 para 82
  fraudes), com apenas 4 falsos alarmes a mais. O F1 foi de 0,8457 para 0,8770, o melhor de todos os testes.
- **XGBoost:** é pouco sensível ao limiar. Baixar para 0,3 pegou 1 fraude a mais ao custo de 2 falsos alarmes.
- **Regressão Logística:** baixar o limiar para 0,3 piorou o modelo (o recall ficou em 0,9184 e os falsos alarmes
  passaram de 1.403 para 3.342). O `class_weight='balanced'` já infla as probabilidades, então aqui o limiar
  precisaria subir, não descer.
- **Conclusão:** o limiar certo depende do modelo e do custo de cada erro. "Baixar para 0,3" funcionou no
  Random Forest, mas não na Regressão Logística.

**Limiar escolhido: 0,3, no Random Forest em Pipeline**, por manter recall de 0,8367 com precisão de 0,9213
e só 7 falsos alarmes. [ ] Se preferir o XGBoost, ajuste esta frase e justifique pelo recall e pela PR-AUC.

## Reamostragem: undersampling x oversampling

Aplicada só ao treino, com avaliação na mesma base de teste.

| Estratégia | Precisão | Recall | F1 | Fraudes | Falsos alarmes |
|---|---|---|---|---|---|
| Undersampling (394 + 394) | 0,0414 | 0,9082 | 0,0792 | 89 | 2.061 |
| SMOTE (fraudes até 10% das normais) | 0,8351 | 0,8265 | 0,8308 | 81 | 16 |

O undersampling teve o maior recall, mas com 2.061 falsos alarmes, porque o modelo aprende com apenas 788
linhas. O SMOTE equilibrou melhor. Por limitação de tempo no Colab, usei SMOTE parcial e um Random Forest
mais leve (50 árvores, `max_depth=15`), então a comparação com o Random Forest original não é perfeitamente
justa. Mesmo assim, o ajuste de limiar (82 fraudes, 7 falsos alarmes) superou as duas estratégias sem alterar
os dados de treino.

## Explicabilidade

**Importância das variáveis (Random Forest):** V14 (cerca de 0,18), V10, V4, V12 e V17 lideram e somam
aproximadamente 60% da importância. Amount e Time ficaram fora do top 10.

**SHAP (XGBoost):** V14, V4, V12 e V10 são as que mais pesam. Valores baixos de V14, V12, V10 e V3 e valores
altos de V4 e V11 empurram a decisão para fraude. Amount teve impacto pequeno. A ordem é coerente com a do
Random Forest.

**Uma transação explicada (waterfall):** V14 (valor -6,17) contribuiu com +6,66 na escala logit, seguida de
V10 (+2,16), levando a saída a 10,45, praticamente 100% de probabilidade de fraude.

**Limitação:** V1 a V28 vêm de PCA, então o SHAP mostra quais componentes pesaram e em que direção, mas não o
que cada uma representa no mundo real.

## O que mudei em relação à Expert

- Testei o limiar em três modelos e mostrei que baixar para 0,3 funciona no Random Forest, mas piora a
  Regressão Logística com peso.
- Levei o Random Forest para o Pipeline com ajuste de limiar (ideia de evolução do desafio).
- Comparei undersampling e SMOTE pelo recall, precisão e falsos alarmes.
- Comparei os modelos também por PR-AUC, além de recall, precisão e F1.
- Fiz a busca de hiperparâmetros numa amostra e retreinei o melhor modelo com todos os dados.

## Como executar

1. Abra o notebook `deteccao_fraude.ipynb` no Google Colab.
2. Execute `Runtime > Run all`.

O dataset é lido direto pelo link, e o arquivo CSV não está neste repositório.
