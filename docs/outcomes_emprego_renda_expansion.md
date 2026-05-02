# Plano de expansao de outcomes: emprego, renda, informalidade e empresas

Este projeto agora ja roda, no formato Economics Letters, os outcomes mensais disponiveis no Novo CAGED:

- vinculos formais ativos (`estoque`);
- admissoes formais;
- desligamentos/demissoes formais;
- saldo liquido de empregos formais;
- taxa de admissao por estoque anterior;
- taxa de desligamento por estoque anterior;
- taxa de rotatividade.

## Outcomes coletados via Base dos Dados/BigQuery

O ambiente local foi autenticado no Google Cloud em 2026-05-01 com o projeto `upa-research`. As consultas e o coletor estao em:

- `sql/rais_income_municipio.sql`;
- `sql/cnpj_births_municipio_monthly.sql`;
- `scripts/03_collect_bd_bigquery.py`.

Os arquivos coletados ficam em:

- `data/processed/rais_income_municipio_panel.csv`;
- `data/processed/cnpj_births_municipio_monthly.csv`.

Para atualizar a coleta:

```powershell
gcloud auth login
gcloud config set project upa-research
python scripts\03_collect_bd_bigquery.py
```

### RAIS anual: renda e vinculos formais

Fonte: RAIS/Base dos Dados. A tabela de microdados de vinculos e grande e particionada por ano e UF. Usar somente `RS`, `SC` e `PR`, selecionar poucas colunas e agregar antes de salvar.

Outcomes sugeridos:

- vinculos formais ativos em 31/12;
- remuneracao media anual;
- remuneracao media de dezembro;
- massa salarial formal;
- horas contratadas;
- composicao por setor, escolaridade, sexo e idade.

Uso econometrico: anual, nao mensal. Deve entrar em exercicio separado, com janela pre/post anual, ou como evidencia complementar, nao misturado diretamente ao CAGED mensal.

### CNPJ: nascimento de empresas

Fonte: tabela CNPJ da Base dos Dados no BigQuery, construida a partir dos dados publicos da Receita Federal. Construir contagens por municipio e mes usando data de inicio de atividade. A tabela contem snapshots mensais; por isso a consulta usa apenas o snapshot mais recente (`data = MAX(data)`) e conta `DISTINCT cnpj`, evitando contar o mesmo estabelecimento repetidas vezes.

Outcomes sugeridos:

- empresas abertas por mes;
- MEIs abertos por mes, se a variavel estiver disponivel no snapshot usado;
- empresas abertas por setor CNAE;
- empresas baixadas/inativas por mes, se houver data de situacao cadastral confiavel.

Uso econometrico: pode ser mensal e, portanto, adequado ao mesmo desenho de controle sintetico, desde que a data de abertura esteja historicamente consistente.

### Informalidade

Fonte primaria: PNAD Continua/Base dos Dados. Restricao importante: a PNAD-C nao e representativa no nivel de municipio comum. Indicadores mensais sao nacionais; indicadores trimestrais sao representativos para Brasil, Grandes Regioes, UFs, algumas regioes metropolitanas, capitais e RIDE Teresina.

Implicacao: informalidade nao deve ser usada como outcome municipal de controle sintetico para Bento Goncalves. Pode entrar como contexto estadual/regional, ou como outcome apenas se o desenho mudar para nivel UF/regiao.

## Decisao para o artigo curto atual

O PDF em `paper/economics_letters_project` usa apenas outcomes mensais municipalmente comparaveis: Bolsa Familia e Novo CAGED. Renda, informalidade e nascimento de empresas ficam para uma extensao de dados para evitar misturar granularidades e frequencias incompativeis.
