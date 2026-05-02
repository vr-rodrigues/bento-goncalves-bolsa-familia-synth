# Fontes

## Politica em Bento Goncalves

- Prefeitura de Bento Goncalves, noticia de 08/10/2025: informa busca ativa, verificacao cadastral e reducao de 2.215 familias em novembro de 2024 para 1.400 em setembro de 2025.
- Correio do Povo, noticia de 31/03/2026: informa reducao de 2.115 familias em novembro de 2024 para 1.296 em marco de 2026 e descreve a busca ativa implantada em novembro de 2024.
- Jornal do Comercio, noticia de 29/11/2024: descreve corte/averiguacao de cadastros unipessoais e encaminhamento para vagas, no programa Bento em Ordem.

## Dados

- MDS/VISDATA: endpoint `https://aplicacoes.mds.gov.br/sagi/servicos/misocial`, variaveis `qtd_familias_beneficiarias_bolsa_familia` e `valor_repassado_bolsa_familia`.
- Novo CAGED/MTE: pagina de Marco de 2026 em `https://www.gov.br/trabalho-e-emprego/pt-br/assuntos/estatisticas-trabalho/novo-caged/2026/marco/pagina-inicial`; a pasta publica de tabelas aponta para a planilha `3-tabelas_Marco de 2026.xlsx`, cuja `Tabela 8` contem evolucao mensal municipal de estoque, admissoes, desligamentos e saldo.
- IBGE Localidades API: diretorio de municipios, UFs e regioes.
- Base dos Dados/BigQuery: preparado para CAGED, PIB municipal, FINBRA e diretorios no projeto `upa-research`.

## Observacao de data

As imagens fornecidas mencionam "2022 e inicio de 2024", mas as fontes oficiais/jornalisticas localizadas indicam novembro de 2024 a marco de 2026. O pipeline usa novembro de 2024 como inicio do tratamento e outubro de 2024 como ultimo mes pre-tratamento.
