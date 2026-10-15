# Roteiro de aprendizado do primeiro modelo

Reserve sessões de até uma hora, ajustando a rotina de candidaturas. Os resultados já executados pelo Codex são o ponto de partida; Gabriel ainda precisa executar, compreender e evoluir o projeto.

## Sessão 1 — Entender os dados e a pergunta

Leia o README e `data.py`. Identifique o que está sendo previsto, quando a previsão seria feita e qual decisão ela apoiaria. Execute o download e carregue os dados. Explique por que `No-show = Yes` significa classe positiva e por que dados clínicos e SMS não foram usados.

Entrega pessoal: escreva em suas palavras o problema, as variáveis e duas limitações. Não copie o texto pronto sem entender.

## Sessão 2 — Datas e vazamento

Execute os testes e o treino. Leia as janelas e as taxas de faltas em `reports/metrics.json`. Explique por que uma consulta agendada antes da fronteira pode ter resultado ainda desconhecido e ser removida do treino. Diferencie separar por agendamento de usar uma divisão aleatória.

Entrega pessoal: desenhe um exemplo de agendamento e consulta que causaria vazamento sem a remoção.

## Sessão 3 — Modelos e métricas

Explique o que faz o modelo de referência e como difere da regressão logística. Leia acurácia, precisão, recall e average precision. Verifique no relatório por que um modelo pode ter acurácia alta e recall zero no limiar de 0,5.

Entrega pessoal: explique a diferença entre classificação, regressão logística e regressão para uma quantidade numérica.

## Sessão 4 — Ranking e capacidade

Leia `top_capacity_metrics`. Explique por que selecionar 20% do teste não demonstra que lembretes evitariam faltas. Faça simulações de capacidade usando o modelo já treinado, sem escolher novo modelo pelo desempenho no teste. Registre o que mudou e as limitações de empates e seleção aleatória esperada.

Entrega pessoal: escreva a diferença entre capturar faltas no ranking e recuperar consultas com uma intervenção.

## Sessão 5 — Uma contribuição sua

Escolha uma melhoria pequena que consiga explicar: teste de uma borda ainda não coberta, documentação de um caso, melhor validação de entrada ou comparação com uma regra definida antes de abrir um novo teste. Não ajustar hiperparâmetros repetidamente no conjunto de teste atual.

Entrega pessoal: faça um commit com problema, decisão, mudança e verificação. Depois pratique uma explicação de cinco minutos sobre o projeto, incluindo o apoio recebido do Codex.
