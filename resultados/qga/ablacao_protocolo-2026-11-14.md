# Protocolo da ablação do caso Souza, Dardenne e Portugal (fixado em 14/11/2026, antes de rodar)

Fixado antes de qualquer execução do script `scripts/qga/ablacao_souza.py`. Alterações depois desta
data entram como emendas datadas no fim deste arquivo. Responde ao parecer da lente 0
(`ArtigoQCalc/parecer-lente0-2026-11-14.md`, objeção 1).

## Objetivo
Demonstrar o controle de ablação do procedimento: decidir se a vantagem relatada pelo preprint
(arXiv:2511.05254 v1) vem do circuito ou do desenho da avaliação (aptidão na média decodificada e
ótimo no centro do domínio).

## Desenho
- Reimplementação: `scripts/qga/souza_v1.py`, escolhas padrão (`PADRAO`), |P| = 50, G = 30, 1024
  medições por avaliação, 2 registros de 8 qubits.
- Braços: quântico (G_qu, com H); análogo clássico ("gêmeo": H trocada por XOR com moeda, Y por X,
  fases descartadas); clássico (G_cl, sem H).
- Funções (domínio usual de largura W): Rastrigin (W = 10,24), Esfera (10,24), Ackley (65,536),
  Griewank (1200), Rosenbrock (4,096). Ótimo o = 0, exceto Rosenbrock (o = 1).
- Domínios: [o − αW, o + (1 − α)W] com α ∈ {0,50 (centrado no ótimo), 0,355, 0,26, 0,16}. Os valores
  de α foram escolhidos para que, no Rastrigin, o centro do domínio fique perto de meio-inteiros
  (deslocamentos de 1,48; 2,46; 3,48), longe dos mínimos locais. Nenhum deslocamento é inteiro.
- Repetições: 50 por braço, função e domínio (3000 execuções). Sementes determinísticas (base
  20261123).
- Saída por execução: curva do melhor f avaliado nas gerações 0 a 30.

## Medidas e testes (fixados)
- Medida: log(f + 1e-9) do melhor f avaliado.
- Margem de equivalência: razão das médias geométricas em [0,80; 1,25] (convenção de
  bioequivalência), isto é, diferença de médias de log em ±ln 1,25 = ±0,2231. Equivalência por TOST
  (dois testes unilaterais de Welch, α = 0,05 cada; equivale ao IC de 90% contido na margem).
- Superioridade: teste unilateral de Welch "quântico melhor (log f menor)", α = 0,05, com correção de
  Holm dentro de cada família.

## Previsões (H_A1 a H_A3)
- **H_A1, assinatura do viés de centro (g0).** Quântico − clássico (log) negativo no domínio
  centrado e não negativo nos três deslocados, em cada função. Família: 5 funções × 4 domínios.
  Relata-se a direção e o IC de 95%.
- **H_A2, sem superioridade final (g30, deslocados).** Em nenhum dos 15 pares (5 funções × 3
  deslocados) o quântico é superior ao clássico (Holm, família de 15). Idem quântico contra análogo.
- **H_A3, equivalência final (g30, deslocados).** Quântico e clássico equivalentes pela margem acima;
  quântico e análogo idem. Relata-se a fração de pares com equivalência demonstrada; os demais são
  "inconclusivos" (nem diferença nem equivalência), não "iguais".
- Também relatado, sem previsão: as mesmas comparações no domínio centrado e em todas as gerações
  (curvas), e onde o análogo fica atrás do quântico.

## Critério de leitura
- Se H_A1 e H_A2 valem: a vantagem relatada é explicada pelo desenho da avaliação.
- Se H_A2 falha em algum par deslocado: relatar como vantagem quântica não explicada naquele par.
- H_A3 inconclusiva não contradiz H_A2; diz que o número de repetições não basta para afirmar empate.

## Emendas
- 14/11/2026: teste de funcionamento do script com 2 repetições por célula, só para conferir que
  roda até o fim; saída apagada sem leitura dos resultados. Nada no protocolo foi alterado.
- 15/11/2026: execução completa terminada em 14/11 (6378 s). Leitura contra as previsões em
  `ArtigoQCalc/plano-artigo-2026-11-14.md`, Seção 4.7.2. A monotonia com o deslocamento foi notada
  depois de ver os dados e é exploratória.
