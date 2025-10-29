---
marp: true
paginate: true
title: Deteccao de Fraudes em Cartoes de Credito
description: Avaliacao e analise do projeto MachineCredit
---

# Deteccao de Fraudes em Cartoes de Credito

Projeto MachineCredit - panorama da avaliação do modelo final em dados desbalanceados.

---

## Metricas Chave

- Acuracia: **0.94**
- Macro Avg F1-Score: **0.93**
- Precisao (fraude): **0.88**
- Recall (fraude): **0.91**

---

## Matriz de Confusao

```
[243  11]
[9    89]
```

- 9 fraudes perdidas (falsos negativos) e 11 alertas falsos.
- Recall acima de 0.90 prioriza a deteccao de fraudes.

---

## Tecnicas Aplicadas

- EDA: exploracão do desbalanceamento e distribuições monetarias.
- Selecionadores: SelectKBest, RFE (logistic) e importancia de Random Forest.
- Balanceamento: comparacão entre SMOTE, Random Undersampling e SMOTEENN.
- Tuning: RandomizedSearchCV (3 folds) em Decision Tree, XGBoost e KNN.
- Ensembles: Voting e Stacking com modelos ajustados.

---

## Impacto do Desbalanceamento

- Dataset com 1.267 transacoes legitimas e 492 fraudulentas.
- Sem reamostragem: recall medio ~0.84.
- Random undersampling elevou o recall base para 0.88.
- Configuracao final (RFE + undersampling + XGBoost) entregou recall 0.91.
- Relatorios e graficos em `outputs/reports` e `outputs/plots`.
