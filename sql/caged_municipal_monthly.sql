-- Outcome de empregabilidade via Base dos Dados.
-- Rode no BigQuery com billing project `upa-research`.
-- Ajuste nomes de campos se a tabela mudar no catalogo da Base dos Dados.

SELECT
  ano,
  mes,
  id_municipio,
  COUNTIF(saldo_movimentacao = 1) AS admissoes,
  COUNTIF(saldo_movimentacao = -1) AS desligamentos,
  SUM(saldo_movimentacao) AS saldo_empregos
FROM `basedosdados.br_me_caged.microdados_movimentacao`
WHERE ano BETWEEN 2023 AND 2026
  AND id_municipio IS NOT NULL
GROUP BY ano, mes, id_municipio
ORDER BY ano, mes, id_municipio;
