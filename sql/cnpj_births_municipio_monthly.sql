-- Monthly firm births by municipality from Receita Federal CNPJ data.
-- Counts establishments by municipality and activity-start month.

SELECT
  CAST(FORMAT_DATE('%Y%m', data_inicio_atividade) AS INT64) AS anomes,
  id_municipio,
  sigla_uf,
  COUNT(DISTINCT cnpj) AS empresas_abertas,
  COUNT(DISTINCT IF(SAFE_CAST(situacao_cadastral AS INT64) = 2, cnpj, NULL)) AS empresas_abertas_ativas_no_snapshot,
  COUNT(DISTINCT IF(cnpj_ordem = '0001', cnpj, NULL)) AS matrizes_abertas
FROM `basedosdados.br_me_cnpj.estabelecimentos`
WHERE
  data = (SELECT MAX(data) FROM `basedosdados.br_me_cnpj.estabelecimentos`)
  AND
  sigla_uf IN ('RS', 'SC', 'PR')
  AND id_municipio IS NOT NULL
  AND data_inicio_atividade BETWEEN DATE('2023-03-01') AND DATE('2026-03-31')
GROUP BY
  anomes,
  id_municipio,
  sigla_uf
ORDER BY
  anomes,
  id_municipio;
