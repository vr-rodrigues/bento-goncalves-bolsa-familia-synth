# Nota de Politica Publica: Bolsa Familia e emprego formal em Bento Goncalves

Projeto de avaliacao com controle sintetico sobre a queda de familias beneficiarias do Bolsa Familia e a evolucao do emprego formal no municipio de Bento Goncalves (RS).

## Pergunta

Depois da politica municipal iniciada em novembro de 2024, o municipio de Bento Goncalves passou a registrar menos familias no Bolsa Familia e mais vinculos formais do que seria esperado a partir de municipios semelhantes da regiao Sul? O exercicio compara o municipio com um contrafactual sintetico construido a partir de municipios nao tratados.

## Estrutura

- `agents/`: papeis dos agentes do projeto.
- `skills/`: habilidades operacionais que os agentes usam.
- `scripts/`: coleta, analise, geracao do artigo e base final.
- `src/bg_bolsa/`: codigo reutilizavel.
- `sql/`: consultas para Base dos Dados/BigQuery.
- `data/final/`: base final leve usada para auditar as estimativas principais.
- `outputs/`: tabelas intermediarias e resultados gerados localmente.
- `paper/economics_letters_project/`: nota final em LaTeX/PDF.

## Execucao rapida

Use Python 3.12 com as dependencias em `pyproject.toml`.

```powershell
python scripts\01_collect_mds_bolsa.py
python scripts\04_collect_caged_tables.py
python scripts\08_build_economics_letters_report.py
python scripts\10_build_final_dataset.py
```

Para empregar Base dos Dados no CAGED/FINBRA/PIB municipal:

```powershell
gcloud auth login
gcloud config set project upa-research
```

Depois rode ou adapte as consultas em `sql/`.

## Artefatos principais

- Artigo final: `paper/economics_letters_project/main.pdf`.
- Fonte LaTeX: `paper/economics_letters_project/main.tex`.
- Base final: `data/final/final_timeseries.csv`, `data/final/final_estimates.csv` e `data/final/final_weights.csv`.

Os dados brutos e intermediarios ficam fora do Git por tamanho e por serem regeneraveis. As consultas em `sql/` documentam as extracoes feitas via Base dos Dados/BigQuery.
