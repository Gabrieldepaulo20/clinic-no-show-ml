# Resultados do primeiro experimento

Experimento retrospectivo com dados públicos de 2016. Resultados calculados pelo pipeline; sem validação operacional em clínicas.

Modelo escolhido por average precision na validação: **logistic_regression**.

| Modelo | AP validação | AP teste | ROC AUC teste | Brier teste | Acurácia a 0,5 | Recall a 0,5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.1040 | 0.1077 | 0.5000 | 0.1066 | 0.8923 | 0.0000 |
| logistic_regression | 0.2239 | 0.2045 | 0.7223 | 0.0899 | 0.8923 | 0.0000 |
| hist_gradient_boosting | 0.2223 | 0.2154 | 0.7234 | 0.0894 | 0.8923 | 0.0000 |

Acurácia usa o limiar ilustrativo 0,5. Escolher AP na validação prioriza avaliação de ranking; não implica superioridade em todas as métricas.

## Simulação de capacidade

Entre os 3320 agendamentos de maior score (20% do teste), 22.8% foram faltas observadas, capturando 42.3% das faltas do teste. Lift retrospectivo sobre seleção aleatória esperada: 2.11x.

Essa simulação não mede consultas recuperadas, ganho financeiro ou efeito de lembretes. Empates no score usam ordem estável; a comparação aleatória é expectativa analítica, não um teste operacional.

## Divisão temporal

| Conjunto | Registros | Taxa de faltas |
| --- | ---: | ---: |
| train | 54889 | 21.0% |
| validation | 12864 | 10.4% |
| test | 16596 | 10.8% |

Início da validação: 2016-05-18T00:00:00+00:00; início do teste: 2016-05-31T00:00:00+00:00.

Removidos do treino: 19638 resultados ainda indisponíveis no início da validação; removidos da validação: 6535 resultados ainda indisponíveis no início do teste.

## Limites

- Historical 2016 sample; no external clinic validation
- Only calendar features; demographics, patient IDs, clinical flags and SMS excluded
- No causal estimate of reminder impact or reduced no-shows
- UTC calendar interpretation; local timezone mapping not verified
- Existing data may include repeated patients across dates; no new-patient holdout
- Scores not externally calibrated; retrospective educational use only
