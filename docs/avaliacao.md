# Avaliação do Modelo Final de Detecção de Fraudes

## Métricas Principais
- **Modelo selecionado:** XGBoost com 15 atributos escolhidos por RFE (Logistic Regression) e Random Undersampling.
- **Acurácia:** 0,94
- **Macro Avg F1-Score:** 0,93
- **Precisão (classe fraude):** 0,88
- **Recall (classe fraude):** 0,91
- **Matriz de confusão (Legítima x Fraude):**
  ```
  [[243  11]
   [  9  89]]
  ```
- **Arquivos de apoio:** `outputs/reports/classification_report_rfe_log_reg_random_undersampling_xgboost_tuned.txt`, `outputs/plots/confusion_matrix_rfe_log_reg_random_undersampling_xgboost_tuned.png`, `outputs/plots/roc_curve_rfe_log_reg_random_undersampling_xgboost_tuned.png`.

## Técnicas Aplicadas e Contribuições
- **Análise Exploratória (EDA):** identificou o desbalanceamento (fraudes ≈ 28 %) e orientou a padronização das variáveis, correlações e distribuição de valores monetários.
- **Seleção de atributos (SelectKBest, RFE, Importância do Random Forest):** os 15 atributos escolhidos via RFE proporcionaram o melhor equilíbrio entre recall e estabilidade para os modelos de maior desempenho.
- **Estratégias de balanceamento:** foram comparados SMOTE, Random Undersampling e SMOTEENN. Em média, o random undersampling elevou a revocação de 0,84 (sem balanceamento) para 0,88 nos modelos base, mostrando maior capacidade de reduzir falsos negativos.
- **Modelos treinados:** Random Forest, XGBoost, SVM, Gradient Boosting, Decision Tree e KNN. O XGBoost destacou-se ao combinar alto recall com boa precisão. Modelos em ensemble (Voting/Stacking) aumentaram a precisão, mas reduziram ligeiramente o recall, e por isso foram mantidos apenas como referência.
- **Tuning de hiperparâmetros:** aplicado aos três modelos com maior recall na fase base (Decision Tree, XGBoost, KNN) usando RandomizedSearchCV com 3 folds estratificados. A busca aleatória com espaço enxuto reduziu o tempo total de execução para ~100 s sem comprometer o ganho de desempenho.

## Impacto do Desbalanceamento e Mitigações
- O conjunto original contém 1 267 transações legítimas e 492 fraudulentas, o que leva modelos sem tratamento a favorecer a classe majoritária.
- Sem balanceamento, a média de recall foi 0,84; técnicas de reamostragem elevaram o valor para até 0,88. A combinação final (RFE + Random Undersampling + XGBoost) manteve recall acima de 0,90, reduzindo falsos negativos para apenas 9 casos no conjunto de teste.
- As métricas macro (F1, precisão e recall) e a matriz de confusão foram essenciais para avaliar o custo do desbalanceamento e validar que o recall não foi inflado à custa de muitos falsos positivos.

## Próximos Passos Recomendados
1. Monitorar o desempenho do modelo em dados recentes para detectar possível degradação de recall ou aumento de falsos positivos.
2. Explorar calibragem de probabilidades ou limiares customizados para ajustar o trade-off entre recall e precisão conforme o custo operacional.
3. Considerar novas variáveis (ex.: dados temporais externos) ou técnicas de detecção semi-supervisionada caso o nível de fraude varie com o tempo.
