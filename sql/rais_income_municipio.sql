-- RAIS annual formal income and employment by municipality.
-- Requires BigQuery billing project access, e.g. `upa-research`.
-- Output is annual and should be used as complementary evidence rather than
-- mixed directly with the monthly CAGED/Bolsa Familia design.

SELECT
  ano,
  id_municipio,
  sigla_uf,
  COUNT(*) AS vinculos_rais,
  COUNTIF(SAFE_CAST(vinculo_ativo_3112 AS INT64) = 1) AS vinculos_ativos_3112,
  AVG(NULLIF(SAFE_CAST(valor_remuneracao_media AS FLOAT64), 0)) AS remuneracao_media,
  AVG(NULLIF(SAFE_CAST(valor_remuneracao_dezembro AS FLOAT64), 0)) AS remuneracao_dezembro_media,
  SUM(COALESCE(SAFE_CAST(valor_remuneracao_media AS FLOAT64), 0)) AS massa_remuneracao_media,
  AVG(NULLIF(SAFE_CAST(quantidade_horas_contratadas AS FLOAT64), 0)) AS horas_contratadas_media
FROM `basedosdados.br_me_rais.microdados_vinculos`
WHERE
  sigla_uf IN ('RS', 'SC', 'PR')
  AND ano BETWEEN 2018 AND 2024
  AND id_municipio IS NOT NULL
GROUP BY
  ano,
  id_municipio,
  sigla_uf
ORDER BY
  ano,
  id_municipio;
