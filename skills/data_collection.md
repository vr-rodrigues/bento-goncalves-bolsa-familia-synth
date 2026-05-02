# Skill: Coleta Municipal

1. Baixar Bolsa Familia pelo MDS/VISDATA usando `scripts/01_collect_mds_bolsa.py`.
2. Baixar Novo CAGED pela planilha oficial usando `scripts/04_collect_caged_tables.py`.
3. Baixar diretorio municipal pelo IBGE API.
4. Quando houver autenticacao Google Cloud, executar SQLs em `sql/` no projeto `upa-research`.
4. Salvar todo bruto em `data/raw/` e toda tabela pronta em `data/processed/`.
5. Nunca misturar codigos de municipio de 6 digitos (MDS) com 7 digitos (IBGE/BD) sem uma chave explicita.
