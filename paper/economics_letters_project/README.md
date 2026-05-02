# Nota em estilo Economics Letters

Esta pasta contem uma nota compacta em duas colunas, em portugues, com figuras montadas a partir de paineis separados. O apendice e um documento separado.

Arquivos principais:

- `main.tex`: fonte do artigo.
- `main.pdf`: relatorio compilado.
- `appendix.tex`: fonte do apendice separado.
- `appendix.pdf`: apendice compilado.
- `figures/`: paineis separados de cada figura.
- `tables/tab_results.tex`: tabela compacta de resultados.

O texto principal destaca Bolsa Familia e estoque formal de vinculos. As consultas complementares estao em `../../sql/` e usam as tabelas publicas da Base dos Dados no BigQuery.

Regenerar a partir da raiz do projeto:

```powershell
python scripts\08_build_economics_letters_report.py
cd paper\economics_letters_project
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode appendix.tex
```
