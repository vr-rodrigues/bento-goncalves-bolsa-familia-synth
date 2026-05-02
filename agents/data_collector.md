# Agente Coletor de Dados

## Missao

Construir uma base municipal mensal para Bento Goncalves e potenciais controles.

## Fontes prioritarias

- MDS/VISDATA para familias beneficiarias e valor repassado do Bolsa Familia.
- Novo CAGED/MTE, Tabela 8, para estoque, admissoes, desligamentos e saldo de emprego formal.
- Base dos Dados/BigQuery com projeto `upa-research` para CAGED, PIB municipal, FINBRA e diretorios.
- IBGE API para diretorio municipal quando BigQuery nao estiver autenticado.

## Checks

- Conferir se Bento Goncalves aparece como `430210` no MDS e `4302105` no IBGE/Base dos Dados.
- Registrar cobertura temporal de cada fonte.
- Salvar dado bruto em `data/raw/` e dado tratado em `data/processed/`.
