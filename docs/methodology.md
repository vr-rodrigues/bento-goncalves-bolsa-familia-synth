# Metodologia

## Desenho

Unidade tratada: Bento Goncalves (MDS `430210`, IBGE `4302105`).

Tratamento: politica municipal de busca ativa, verificacao cadastral e encaminhamento para vagas de trabalho, documentada a partir de novembro de 2024.

Janela:

- Pre-tratamento: marco de 2023 a outubro de 2024.
- Pos-tratamento: novembro de 2024 ate o ultimo mes disponivel.

## Matching

Antes do controle sintetico, o pipeline restringe o pool de doadores aos municipios mais proximos de Bento em:

- nivel medio pre de familias beneficiarias;
- beneficio medio pre;
- inclinacao pre da serie em log;
- variacao percentual pre;
- pontos selecionados da trajetoria pre.
- defasagens mensais de `y` no pre-tratamento, normalizadas por outubro de 2024.

O procedimento roda em dois pools:

- Brasil inteiro;
- somente municipios do Rio Grande do Sul.

## Controle sintetico

O outcome e normalizado para cada municipio como indice, com outubro de 2024 igual a 100. O algoritmo escolhe pesos nao negativos, somando 1, que minimizam o erro quadratico medio entre Bento e os doadores em toda a trajetoria pre-tratamento. Na pratica, os lags de `y` entram duas vezes: primeiro para escolher doadores parecidos no matching e depois como preditores no ajuste dos pesos do synth.

O efeito em indice e convertido para familias usando o numero de familias beneficiarias de Bento em outubro de 2024.

## Placebos

Cada municipio no pool pareado e tratado como se fosse Bento, usando os demais como doadores. A estatistica principal e a razao entre RMSPE pos e RMSPE pre. O p-valor empirico compara a razao de Bento com a distribuicao placebo.

## Testes de robustez implementados

- Placebos em espaco/permutation test nos doadores pareados.
- Razao RMSPE pos/pre para avaliar se o gap pos-tratamento e incomum em relacao ao fit pre.
- Placebo temporal com falso tratamento em maio de 2024 e falso pos ate outubro de 2024.
- Leave-one-out retirando doadores com peso relevante.
- Rodadas separadas para pool nacional e pool restrito ao Rio Grande do Sul.

## Outcomes

- Familias beneficiarias do Bolsa Familia.
- Estoque de empregos formais.
- Admissoes formais mensais.
- Desligamentos formais mensais.
