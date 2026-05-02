# Replicação em Python dos ICs de Firpo e Possebom

Este projeto agora usa uma porta em Python da função `SCM.CS` do suplemento de Firpo e Possebom, guardado em `scripts/suppl_jci-2016-0026supp/function_SCM-CS_v07.R`.

## O que a função dos autores faz

A rotina recebe:

- `Ymat`: matriz de resultados observados, com períodos nas linhas e unidades nas colunas;
- `weightsmat`: matriz de pesos sintéticos, uma coluna para cada unidade tratada/placebo;
- `treated`: coluna da unidade tratada;
- `T0`: último período pré-intervenção;
- `type`: classe do efeito pós-tratamento, `constant` ou `linear`;
- `significance`: nível de significância;
- `phi` e `v`: parâmetros de sensibilidade. Para o caso padrão, `phi = 0` e `v = 0`, então todas as unidades são igualmente prováveis de receber tratamento.

O algoritmo não constrói uma banda como quantil simples dos placebos. Ele inverte testes placebo:

1. Calcula uma estimativa inicial do parâmetro de efeito. Para `linear`, ela é o efeito no último período dividido pelo número de períodos pós-tratamento.
2. Propõe um efeito nulo `nh`: zero no pré e constante ou linear no pós.
3. Para cada unidade, ajusta a matriz de resultados sob esse nulo. Quando uma unidade placebo é tratada, a unidade originalmente tratada entra como controle com o efeito nulo subtraído.
4. Calcula a estatística RMSPE pós/pré para a unidade tratada e para todos os placebos.
5. Calcula o p-valor pelo posto da estatística tratada na distribuição das estatísticas placebo.
6. Expande e refina os limites superior e inferior até encontrar os valores nulos não rejeitados no nível escolhido.

O retorno é um conjunto de confiança para uma classe de funções de efeito, não uma banda placebo centrada mecanicamente na diferença estimada.

## Porta em Python

A implementação está em:

- `src/bg_bolsa/firpo_possebom.py`

Ela replica a estrutura da função R:

- `scm_confidence_set(...)`: porta direta da lógica `SCM.CS`;
- `build_scm_matrices(...)`: monta `Ymat` e `weightsmat` usando o estimador sintético do projeto;
- `confidence_set_for_panel(...)`: wrapper usado pelo relatório, com `alpha = 0.10`, `effect_type = "linear"` e média móvel de 3 meses para alinhar o IC ao gráfico.

Como o estimador do projeto usa outcomes indexados quando `scale_mode = "index"`, o IC é calculado na escala de índice e depois convertido para a escala do efeito reportado usando o valor de Bento Gonçalves no mês base. Isso mantém consistência entre pesos, estatística RMSPE e figura.

## Aplicação no relatório

O script `scripts/08_build_economics_letters_report.py` agora:

- calcula e salva `outputs/economics_letters/*_firpo_possebom_ci.csv`;
- usa esses limites no painel `Efeito e IC 90%`;
- atualiza o texto para descrever os ICs como conjuntos de confiança de Firpo e Possebom portados para Python, não como uma banda própria “no espírito” do artigo.

Nos casos em que a própria rotina retorna `Empty confidence set for the requested class of effects`, o CSV guarda esse diagnóstico em `fp_error` e o gráfico não usa uma banda alternativa. No relatório atual, os quatro desfechos principais com pool Sul têm IC linear não vazio; alguns desfechos do apêndice/robustez retornam conjunto vazio.
