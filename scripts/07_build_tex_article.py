from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
TABLES_IN = ROOT / "outputs" / "tables"
PAPER = ROOT / "paper" / "overleaf_project"


OUTCOME_LABELS = {
    "bolsa_familia": "Bolsa Familia",
    "emprego_estoque": "Emprego formal: estoque",
    "emprego_admissoes": "Emprego formal: admissoes",
    "emprego_desligamentos": "Emprego formal: desligamentos",
}


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def fmt0(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):,.0f}".replace(",", ".")


def tex_escape(value: object) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def write_result_table(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    tex_escape(OUTCOME_LABELS[row["outcome"]]),
                    tex_escape("Brasil" if row["pool"] == "national" else "RS"),
                    fmt(row["pre_rmspe_index"]),
                    fmt(row["post_pre_rmspe_ratio"]),
                    fmt(row["p_value_space_placebo"], 3),
                    fmt(row["mean_post_effect_value"], 1),
                    fmt0(row["last_actual_value"]),
                    fmt0(row["last_synth_value"]),
                    fmt0(row["last_effect_value"]),
                ]
            )
            + r" \\"
        )
    content = r"""\begin{threeparttable}
\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{l c r r r r r r r}
\toprule
Outcome & Pool & RMSPE pre & Pos/pre & $p$ placebo & Efeito medio & Observado & Synth & Efeito \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\tabnotes{\textit{Notes:} O efeito medio e o efeito do ultimo mes estao em unidades originais do outcome. O $p$ placebo compara a razao RMSPE pos/pre de Bento Goncalves com a distribuicao de placebos em espaco no pool pareado.}
\end{threeparttable}
"""
    path.write_text(content, encoding="utf-8")


def write_test_table(summary: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            " & ".join(
                [
                    tex_escape(OUTCOME_LABELS[row["outcome"]]),
                    tex_escape("Brasil" if row["pool"] == "national" else "RS"),
                    fmt(row["fake_time_placebo_mean_effect_index"]),
                    fmt(row["fake_time_placebo_last_effect_index"]),
                    fmt0(row["leave_one_out_last_effect_min"]),
                    fmt0(row["leave_one_out_last_effect_max"]),
                ]
            )
            + r" \\"
        )
    content = r"""\begin{threeparttable}
\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{l c r r r r}
\toprule
Outcome & Pool & Placebo temporal: media & Placebo temporal: ultimo & LOO min & LOO max \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\tabnotes{\textit{Notes:} Placebo temporal usa falso tratamento em maio de 2024 e falso pos-tratamento ate outubro de 2024. LOO indica a faixa do efeito no ultimo mes ao retirar doadores com peso relevante.}
\end{threeparttable}
"""
    path.write_text(content, encoding="utf-8")


def write_data_table(path: Path) -> None:
    content = r"""\begin{threeparttable}
\small
\begin{tabularx}{\linewidth}{l l X}
\toprule
Fonte & Nivel & Variaveis usadas \\
\midrule
MDS/VISDATA & Municipio-mes & Familias beneficiarias e valor repassado do Bolsa Familia \\
Novo CAGED/MTE, Tabela 8 & Municipio-mes & Estoque, admissoes, desligamentos e saldo de emprego formal \\
IBGE Localidades & Municipio & Nome do municipio, UF, regiao e microrregiao \\
\bottomrule
\end{tabularx}
\tabnotes{\textit{Notes:} O codigo municipal do MDS usa seis digitos. O projeto mantem chave explicita para compatibilizar com diretorios municipais.}
\end{threeparttable}
"""
    path.write_text(content, encoding="utf-8")


def draw_synth_figure(timeseries_path: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(timeseries_path)
    width, height = letter
    c = canvas.Canvas(str(out_path), pagesize=letter)
    margin = 54
    chart_w = width - 2 * margin
    chart_h = 310
    x0 = margin
    y0 = 130

    actual = df["actual_index"].astype(float).tolist()
    synth = df["synth_index"].astype(float).tolist()
    ymin = min(min(actual), min(synth)) - 5
    ymax = max(max(actual), max(synth)) + 5
    denom = max(ymax - ymin, 1e-9)

    def xy(idx: int, value: float) -> tuple[float, float]:
        x = x0 + idx * chart_w / max(len(df) - 1, 1)
        y = y0 + (value - ymin) * chart_h / denom
        return x, y

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - 70, title)
    c.setFont("Helvetica", 10)
    c.drawString(margin, height - 88, "Indice: outubro de 2024 = 100")

    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.line(x0, y0, x0 + chart_w, y0)
    c.line(x0, y0, x0, y0 + chart_h)

    matches = df.index[df["anomes"] == 202411].tolist()
    if matches:
        x, _ = xy(matches[0], ymin)
        c.setStrokeColor(colors.HexColor("#777777"))
        c.setDash(4, 4)
        c.line(x, y0, x, y0 + chart_h)
        c.setDash()
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(x + 4, y0 + chart_h - 14, "tratamento")

    for series, color in [(synth, colors.HexColor("#3578b8")), (actual, colors.HexColor("#d14b3f"))]:
        c.setStrokeColor(color)
        c.setLineWidth(1.8)
        for idx in range(len(series) - 1):
            c.line(*xy(idx, series[idx]), *xy(idx + 1, series[idx + 1]))

    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#d14b3f"))
    c.drawString(x0 + chart_w - 150, y0 + chart_h + 16, "Bento Goncalves")
    c.setFillColor(colors.HexColor("#3578b8"))
    c.drawString(x0 + chart_w - 150, y0 + chart_h + 2, "Controle sintetico")

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 9)
    c.drawString(x0, y0 - 18, str(int(df.iloc[0]["anomes"])))
    c.drawCentredString(x0 + chart_w / 2, y0 - 18, str(int(df.iloc[len(df) // 2]["anomes"])))
    c.drawRightString(x0 + chart_w, y0 - 18, str(int(df.iloc[-1]["anomes"])))
    c.save()


def write_references(path: Path) -> None:
    content = r"""@article{abadie2010synthetic,
  author = {Abadie, Alberto and Diamond, Alexis and Hainmueller, Jens},
  title = {Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program},
  journal = {Journal of the American Statistical Association},
  year = {2010},
  volume = {105},
  number = {490},
  pages = {493--505}
}

@article{abadie2015comparative,
  author = {Abadie, Alberto and Diamond, Alexis and Hainmueller, Jens},
  title = {Comparative Politics and the Synthetic Control Method},
  journal = {American Journal of Political Science},
  year = {2015},
  volume = {59},
  number = {2},
  pages = {495--510}
}

@article{abadie2021using,
  author = {Abadie, Alberto},
  title = {Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects},
  journal = {Journal of Economic Literature},
  year = {2021},
  volume = {59},
  number = {2},
  pages = {391--425}
}

@misc{mdsvisdata,
  author = {{Ministerio do Desenvolvimento e Assistencia Social}},
  title = {MDS/VISDATA: series municipais do Bolsa Familia},
  year = {2026},
  howpublished = {\url{https://aplicacoes.mds.gov.br/sagi/servicos/misocial}},
  note = {Acesso em 1 maio 2026}
}

@misc{mtecaged2026,
  author = {{Ministerio do Trabalho e Emprego}},
  title = {Novo CAGED: Marco de 2026, Tabelas},
  year = {2026},
  howpublished = {\url{https://www.gov.br/trabalho-e-emprego/pt-br/assuntos/estatisticas-trabalho/novo-caged/2026/marco/pagina-inicial}},
  note = {Acesso em 1 maio 2026}
}
"""
    path.write_text(content, encoding="utf-8")


def write_main_tex(path: Path) -> None:
    content = r"""\documentclass[12pt]{article}

\usepackage[margin=0.85in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{natbib}
\usepackage{float}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{subcaption}
\usepackage{tabularx}
\usepackage{threeparttable}
\usepackage{longtable}
\usepackage{pdflscape}

\hypersetup{colorlinks=true, linkcolor=black, citecolor=red, urlcolor=black}
\captionsetup[table]{font=footnotesize,labelfont=bf,justification=centering,singlelinecheck=false}
\bibliographystyle{plainnat}
\setcitestyle{authoryear,round}
\graphicspath{{figures/}}

\linespread{1.08}
\setlength{\parskip}{3pt}

\newcommand{\tabnotes}[1]{\par\vspace{3pt}\noindent\begin{minipage}{\linewidth}\footnotesize #1\end{minipage}}

\title{\textbf{A Politica Municipal de Bento Goncalves e a Queda no Bolsa Familia:\\ Evidencia por Controle Sintetico}}
\author{
    Victor Rangel\thanks{Projeto exploratorio de avaliacao econometrica.}\\
    \small Insper
}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
\noindent Este artigo estima o efeito associado a uma estrategia municipal de busca ativa, verificacao cadastral e encaminhamento ao mercado de trabalho em Bento Goncalves (RS). O desenho empirico combina pareamento em caracteristicas observaveis e defasagens da variavel dependente com controle sintetico. O tratamento e datado em novembro de 2024, com periodo pre-tratamento de marco de 2023 a outubro de 2024 e pos-tratamento de novembro de 2024 a marco de 2026. A evidencia principal indica queda adicional de beneficiarios do Bolsa Familia: no pool restrito ao Rio Grande do Sul, Bento apresenta 1.296 familias beneficiarias em marco de 2026 contra 1.650 no contrafactual sintetico, uma diferenca de -354 familias. Testes placebo sugerem significancia empirica moderada para esse outcome ($p=0{,}065$). Para emprego formal, os efeitos sao pequenos ou instaveis. Os resultados favorecem a interpretacao de queda adicional do programa, mas nao permitem concluir que houve ganho robusto de empregabilidade formal.

\vspace{4pt}
\noindent\textit{JEL}: C23, C54, H53, I38, J21 \quad \textit{Keywords}: Bolsa Familia; controle sintetico; municipios; mercado de trabalho; Bento Goncalves
\end{abstract}

\clearpage

\section{Introducao}

A avaliacao de politicas locais de assistencia social enfrenta um problema classico de identificacao: municipios adotam intervencoes em momentos nao aleatorios e, simultaneamente, estao expostos a choques nacionais, revisoes cadastrais federais e dinamicas locais de mercado de trabalho. Bento Goncalves recebeu atencao publica por relatar forte reducao no numero de familias beneficiarias do Bolsa Familia apos uma estrategia municipal que combinou aproximacao entre beneficiarios e empresas, apoio a curriculos, orientacao profissional e medidas de verificacao cadastral.

A pergunta empirica deste artigo e se a queda observada pode ser interpretada como efeito causal da politica municipal, ou se apenas reflete tendencias comuns a municipios comparaveis. Para isso, constroi-se um contrafactual sintetico de Bento Goncalves a partir de municipios doadores. O exercicio e feito em dois pools: Brasil inteiro e apenas Rio Grande do Sul. A analise principal usa como outcome o numero de familias beneficiarias do Bolsa Familia; outcomes secundarios capturam emprego formal pelo Novo CAGED: estoque, admissoes e desligamentos.

\section{Contexto institucional}

As fontes oficiais e jornalisticas consultadas descrevem uma politica municipal iniciada em novembro de 2024, voltada a revisao ou regularizacao cadastral e encaminhamento de beneficiarios a vagas formais. Esse ponto e importante porque algumas pecas de comunicacao mencionam janelas temporais diferentes. Neste artigo, o tratamento operacional e novembro de 2024, e outubro de 2024 e o ultimo mes pre-tratamento.

Uma reducao no numero de beneficiarios nao e, por si so, equivalente a reducao de pobreza. Ela pode decorrer de aumento de renda, alteracao de composicao familiar, revisao cadastral, migracao, bloqueios, cancelamentos ou mudancas administrativas. Por isso, os outcomes de emprego formal sao tratados como teste substantivo adicional, nao como corroboracao automatica.

\section{Dados}

O painel municipal mensal combina duas fontes principais. Para Bolsa Familia, usa-se o endpoint publico MDS/VISDATA, com numero de familias beneficiarias e valor repassado por municipio e mes \citep{mdsvisdata}. Para emprego formal, usa-se a planilha oficial do Novo CAGED/MTE, Tabela 8, que contem evolucao mensal municipal de estoque, admissoes, desligamentos e saldo \citep{mtecaged2026}. Os codigos municipais do MDS usam seis digitos; a correspondencia com o diretorio municipal do IBGE e feita por uma chave explicita.

A unidade tratada e Bento Goncalves, codigo MDS 430210. O periodo pre-tratamento vai de 202303 a 202410. O periodo pos-tratamento vai de 202411 a 202603.

\begin{table}[H]
\centering
\caption{Fontes de dados}
\label{tab:data_sources}
\input{tables/tab_data_sources.tex}
\end{table}

\section{Estrategia empirica}

A estrategia tem duas etapas. Primeiro, restringe-se o conjunto de doadores por pareamento em observaveis e trajetoria pre-tratamento. As covariadas incluem nivel medio pre, inclinacao pre, variacao percentual pre, volatilidade pre e defasagens mensais da propria variavel dependente, normalizadas por outubro de 2024. Segundo, estima-se um controle sintetico no pool pareado, seguindo a logica de \citet{abadie2010synthetic,abadie2015comparative,abadie2021using}.

Formalmente, seja $Y_{it}$ o outcome do municipio $i$ no mes $t$. Para Bento Goncalves, a trajetoria contrafactual e aproximada por uma combinacao convexa dos municipios doadores:
\[
\widehat{Y}_{1t}(0)=\sum_{j=2}^{J+1}w_jY_{jt}, \quad w_j\geq 0, \quad \sum_j w_j=1.
\]
Os pesos sao escolhidos para minimizar o erro quadratico medio no periodo pre-tratamento:
\[
\min_w \sum_{t\in T_0}\left(Y_{1t}-\sum_j w_jY_{jt}\right)^2.
\]
O efeito estimado no pos-tratamento e:
\[
\widehat{\alpha}_{1t}=Y_{1t}-\widehat{Y}_{1t}(0).
\]
Todos os outcomes sao normalizados como indice com outubro de 2024 igual a 100 para ajuste dos pesos. Os efeitos finais sao reconvertidos para unidades originais.

\section{Inferencia e testes}

A inferencia segue praticas usuais de controle sintetico. Primeiro, roda-se placebo em espaco: cada municipio do pool pareado e tratado como se tivesse recebido a politica, e calcula-se sua razao RMSPE pos/pre. O $p$-valor empirico compara a razao de Bento com a distribuicao placebo. Segundo, roda-se um placebo temporal, com falso tratamento em maio de 2024 e falso pos-tratamento ate outubro de 2024. Terceiro, faz-se leave-one-out, retirando doadores com peso relevante para verificar se o resultado depende de uma unica unidade.

\section{Resultados}

A Tabela~\ref{tab:main_results} resume os resultados principais. A evidencia mais forte aparece no outcome de familias beneficiarias do Bolsa Familia. No pool restrito ao Rio Grande do Sul, o controle sintetico projeta 1.650 familias beneficiarias em marco de 2026, contra 1.296 observadas. O efeito estimado e de -354 familias no ultimo mes e media de -281 familias por mes no pos-tratamento. A razao RMSPE pos/pre e 6,35, com $p$-valor empirico de 0,065.

\begin{table}[H]
\centering
\caption{Resultados principais}
\label{tab:main_results}
\input{tables/tab_main_results.tex}
\end{table}

Para emprego formal, os resultados sao mais fracos. O estoque formal apresenta efeito positivo pequeno no pool RS: 85 empregos no ultimo mes, com $p$-valor de 0,290. Para admissoes, o pool RS apresenta $p$-valor baixo, mas o efeito em unidades e pequeno e o leave-one-out cruza sinais. Para desligamentos, ha aumento no ultimo mes, mas sem suporte placebo convincente. Assim, a leitura mais conservadora e que o efeito mais robusto aparece na saida do Bolsa Familia, enquanto os dados de emprego formal nao confirmam uma melhora causal forte de mercado de trabalho.

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{fig_synth_bolsa_rs.pdf}
\caption{Bolsa Familia: Bento Goncalves e controle sintetico, pool RS}
\label{fig:bf_rs}
\tabnotes{\textit{Notes:} O outcome e indexado para outubro de 2024 igual a 100. A linha vertical marca novembro de 2024.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.82\textwidth]{fig_synth_emprego_estoque_rs.pdf}
\caption{Estoque de emprego formal: Bento Goncalves e controle sintetico, pool RS}
\label{fig:stock_rs}
\tabnotes{\textit{Notes:} O outcome e indexado para outubro de 2024 igual a 100. A linha vertical marca novembro de 2024.}
\end{figure}

\section{Robustez}

A Tabela~\ref{tab:robustness} apresenta o placebo temporal e os resultados leave-one-out. Para Bolsa Familia, o efeito no ultimo mes permanece negativo ao retirar doadores com peso relevante. No pool RS, a faixa leave-one-out vai de -364 a -342 familias. Para emprego formal, a estabilidade e menor. O estoque formal no pool RS varia de aproximadamente zero a 151 empregos; admissoes cruzam sinais; desligamentos sao sensiveis a doadores.

\begin{table}[H]
\centering
\caption{Placebo temporal e leave-one-out}
\label{tab:robustness}
\input{tables/tab_robustness_tests.tex}
\end{table}

\section{Ameacas a identificacao}

Quatro ameacas merecem destaque. Primeiro, revisoes federais do Cadastro Unico podem gerar quedas no Bolsa Familia independentemente da politica municipal. Segundo, a calamidade no Rio Grande do Sul em 2024 pode afetar tanto mercado de trabalho quanto transferencias, embora o pool RS reduza parcialmente essa preocupacao. Terceiro, a politica combina componentes distintos: fiscalizacao cadastral, orientacao, encaminhamento a vagas e possiveis mudancas administrativas. O desenho estima um efeito reduzido da estrategia como pacote, nao de cada componente. Quarto, saida do beneficio nao implica necessariamente melhora de renda ou bem-estar.

\section{Conclusao}

O exercicio de controle sintetico sugere que Bento Goncalves teve queda adicional no numero de familias beneficiarias do Bolsa Familia apos novembro de 2024. A magnitude no pool RS e de aproximadamente 354 familias a menos em marco de 2026, com evidencia placebo moderada. Entretanto, os outcomes de emprego formal nao mostram efeito robusto comparavel. A conclusao substantiva e que ha evidencia sugestiva de efeito sobre permanencia no programa, mas ainda nao evidencia forte de que a politica tenha aumentado empregabilidade formal em magnitude suficiente para explicar toda a queda.

\vspace{8pt}
{\small\linespread{1.0}\selectfont
\bibliography{references}
}

\end{document}
"""
    path.write_text(content, encoding="utf-8")


def write_title_page(path: Path) -> None:
    content = r"""\documentclass[12pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{hyperref}

\begin{document}

\begin{center}
{\Large \textbf{A Politica Municipal de Bento Goncalves e a Queda no Bolsa Familia:\\ Evidencia por Controle Sintetico}}\\[1.5em]
{\large Separate Title Page for Submission}\\[2em]
\end{center}

\noindent\textbf{Author}\\
Victor Rangel\\
Insper\\
E-mail: \href{mailto:victorrsr@al.insper.edu.br}{victorrsr@al.insper.edu.br}

\vspace{1em}
\noindent\textbf{Funding}\\
No external funding is declared at this stage.

\vspace{1em}
\noindent\textbf{Conflicts of Interest}\\
The author declares no conflicts of interest.

\vspace{1em}
\noindent\textbf{Data Availability}\\
The Bolsa Familia data come from public MDS/VISDATA municipal series. The employment data come from the official Novo CAGED municipal tables. Replication scripts and generated outputs are documented in the project folder.

\vspace{1em}
\noindent\textbf{Use of AI Tools}\\
AI tools were used for code assistance, drafting support, and editorial revision. The author reviewed and remains responsible for all empirical choices, text, tables, figures, citations, and conclusions.

\end{document}
"""
    path.write_text(content, encoding="utf-8")


def write_readme(path: Path) -> None:
    content = r"""# Overleaf project

This directory is the LaTeX/Overleaf version of the Bento Goncalves Bolsa Familia synthetic-control paper.

Main files:

- `main.tex`: manuscript.
- `references.bib`: bibliography.
- `tables/`: LaTeX tables generated from `outputs/tables/synth_tests_summary.csv`.
- `figures/`: PDF figures generated from synthetic-control time series.

Regenerate:

```powershell
python scripts\07_build_tex_article.py
```

Compile locally:

```powershell
cd paper\overleaf_project
latexmk -pdf main.tex
```

To upload to Overleaf, use the project files at this directory root: `main.tex`, `references.bib`, `figures/`, and `tables/`.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(TABLES_IN / "synth_tests_summary.csv")
    write_data_table(PAPER / "tables" / "tab_data_sources.tex")
    write_result_table(summary, PAPER / "tables" / "tab_main_results.tex")
    write_test_table(summary, PAPER / "tables" / "tab_robustness_tests.tex")
    draw_synth_figure(
        TABLES_IN / "synth_bolsa_familia_rs_timeseries.csv",
        PAPER / "figures" / "fig_synth_bolsa_rs.pdf",
        "Bolsa Familia: Bento Goncalves vs controle sintetico (RS)",
    )
    draw_synth_figure(
        TABLES_IN / "synth_emprego_estoque_rs_timeseries.csv",
        PAPER / "figures" / "fig_synth_emprego_estoque_rs.pdf",
        "Estoque formal: Bento Goncalves vs controle sintetico (RS)",
    )
    write_references(PAPER / "references.bib")
    write_main_tex(PAPER / "main.tex")
    write_title_page(PAPER / "title_page_author_info.tex")
    write_readme(PAPER / "README.md")
    print(PAPER)


if __name__ == "__main__":
    main()
