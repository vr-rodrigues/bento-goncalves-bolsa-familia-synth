-- Gastos municipais por funcao via FINBRA/Base dos Dados.
-- A taxonomia da FINBRA muda por tabela/ano; valide campos no catalogo antes de rodar em producao.

SELECT
  ano,
  id_municipio,
  conta,
  estagio_bd,
  SUM(valor) AS valor
FROM `basedosdados.br_tesouro_finbra.despesas_orcamentarias`
WHERE ano BETWEEN 2021 AND 2024
  AND id_municipio IS NOT NULL
GROUP BY ano, id_municipio, conta, estagio_bd
ORDER BY ano, id_municipio, conta, estagio_bd;
