-- Caracteristicas municipais via Base dos Dados.
-- Rode no BigQuery com billing project `upa-research`.

WITH municipio AS (
  SELECT
    id_municipio,
    nome AS municipio,
    sigla_uf,
    id_microrregiao,
    id_mesorregiao
  FROM `basedosdados.br_bd_diretorios_brasil.municipio`
),
pib AS (
  SELECT
    id_municipio,
    ano,
    pib,
    impostos_liquidos,
    va_agropecuaria,
    va_industria,
    va_servicos,
    va_adespss
  FROM `basedosdados.br_ibge_pib.municipio`
  WHERE ano = 2021
),
pop AS (
  SELECT
    id_municipio,
    populacao
  FROM `basedosdados.br_ibge_populacao.municipio`
  WHERE ano = 2022
)
SELECT
  m.*,
  pop.populacao,
  pib.pib,
  SAFE_DIVIDE(pib.pib, pop.populacao) AS pib_per_capita,
  pib.impostos_liquidos,
  pib.va_agropecuaria,
  pib.va_industria,
  pib.va_servicos,
  pib.va_adespss
FROM municipio AS m
LEFT JOIN pop USING (id_municipio)
LEFT JOIN pib USING (id_municipio);
