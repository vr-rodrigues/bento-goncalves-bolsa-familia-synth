from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "reports"

OUTCOME_LABELS = {
    "bolsa_familia": "Familias beneficiarias do Bolsa Familia",
    "emprego_estoque": "Estoque de empregos formais",
    "emprego_admissoes": "Admissoes formais mensais",
    "emprego_desligamentos": "Desligamentos formais mensais",
}


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt0(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.0f}".replace(",", ".")


def build_markdown(summary: pd.DataFrame) -> str:
    bf_rs = summary[(summary["outcome"] == "bolsa_familia") & (summary["pool"] == "rs")].iloc[0]
    stock_rs = summary[
        (summary["outcome"] == "emprego_estoque") & (summary["pool"] == "rs")
    ].iloc[0]

    lines: list[str] = []
    lines.extend(
        [
            "# A politica municipal de Bento Goncalves e a queda no Bolsa Familia: evidencia por controle sintetico",
            "",
            "**Resumo.** Este artigo estima o efeito associado a uma estrategia municipal de busca ativa, verificacao cadastral e encaminhamento ao mercado de trabalho em Bento Goncalves (RS). O desenho empirico combina pareamento em caracteristicas observaveis e defasagens da variavel dependente com controle sintetico. O tratamento e datado em novembro de 2024, com periodo pre-tratamento de marco de 2023 a outubro de 2024 e pos-tratamento de novembro de 2024 a marco de 2026. A evidencia principal indica queda adicional de beneficiarios do Bolsa Familia: no pool restrito ao Rio Grande do Sul, Bento apresenta 1.296 familias beneficiarias em marco de 2026 contra 1.650 no contrafactual sintetico, uma diferenca de -354 familias. Testes placebo sugerem significancia empirica moderada para esse outcome (p = 0,065). Para emprego formal, os efeitos sao pequenos ou instaveis: o estoque formal aumenta apenas 85 postos no pool RS e nao se destaca na distribuicao placebo. Os resultados, portanto, favorecem a interpretacao de queda adicional do programa, mas nao permitem concluir que houve ganho robusto de empregabilidade formal.",
            "",
            "**Palavras-chave:** Bolsa Familia; controle sintetico; municipios; mercado de trabalho; Bento Goncalves.",
            "",
            "**JEL:** C23; C54; H53; I38; J21.",
            "",
            "## 1. Introducao",
            "",
            "A avaliacao de politicas locais de assistencia social enfrenta um problema classico de identificacao: municipios adotam intervencoes em momentos nao aleatorios e, ao mesmo tempo, estao expostos a choques nacionais, revisoes cadastrais federais e dinamicas locais de mercado de trabalho. Bento Goncalves recebeu atencao publica por relatar forte reducao no numero de familias beneficiarias do Bolsa Familia apos uma estrategia municipal que combinou aproximacao entre beneficiarios e empresas, apoio a curriculos, orientacao profissional e medidas de verificacao cadastral.",
            "",
            "A pergunta empirica deste artigo e se a queda observada pode ser interpretada como efeito causal da politica municipal, ou se apenas reflete tendencias comuns a municipios comparaveis. Para isso, constroi-se um contrafactual sintetico de Bento Goncalves a partir de municipios doadores. O exercicio e feito em dois pools: Brasil inteiro e apenas Rio Grande do Sul. A analise principal usa como outcome o numero de familias beneficiarias do Bolsa Familia; outcomes secundarios capturam emprego formal pelo Novo CAGED: estoque, admissoes e desligamentos.",
            "",
            "## 2. Contexto institucional",
            "",
            "As fontes jornalisticas e oficiais consultadas descrevem uma politica municipal iniciada em novembro de 2024, voltada a revisao/regularizacao cadastral e encaminhamento de beneficiarios a vagas formais. Esse ponto e importante porque algumas pecas de comunicacao mencionam janelas temporais diferentes. Neste artigo, o tratamento operacional e novembro de 2024, e outubro de 2024 e o ultimo mes pre-tratamento.",
            "",
            "Uma reducao no numero de beneficiarios nao e, por si so, equivalente a reducao de pobreza. Ela pode decorrer de aumento de renda, alteracao de composicao familiar, revisao cadastral, migracao, bloqueios/cancelamentos ou mudancas administrativas. Por isso, os outcomes de emprego formal sao tratados como teste substantivo adicional, nao como mera corroboracao automatica.",
            "",
            "## 3. Dados",
            "",
            "O painel municipal mensal combina duas fontes principais. Para Bolsa Familia, usa-se o endpoint publico MDS/VISDATA, com numero de familias beneficiarias e valor repassado por municipio e mes. Para emprego formal, usa-se a planilha oficial do Novo CAGED/MTE, Tabela 8, que contem evolucao mensal municipal de estoque, admissoes, desligamentos e saldo. Os codigos municipais do MDS usam seis digitos; a correspondencia com o diretorio municipal do IBGE e feita por uma chave explicita.",
            "",
            "A unidade tratada e Bento Goncalves, codigo MDS 430210. O periodo pre-tratamento vai de 202303 a 202410. O periodo pos-tratamento vai de 202411 a 202603.",
            "",
            "## 4. Estrategia empirica",
            "",
            "A estrategia tem duas etapas. Primeiro, restringe-se o conjunto de doadores por pareamento em observaveis e trajetoria pre-tratamento. As covariadas incluem nivel medio pre, inclinacao pre, variacao percentual pre, volatilidade pre e defasagens mensais da propria variavel dependente, normalizadas por outubro de 2024. Segundo, estima-se um controle sintetico no pool pareado.",
            "",
            "Formalmente, seja Y_it o outcome do municipio i no mes t. Para Bento Goncalves, a trajetoria contrafactual e aproximada por uma combinacao convexa dos municipios doadores:",
            "",
            "$$ \\hat{Y}_{1t}(0) = \\sum_{j=2}^{J+1} w_j Y_{jt}, \\quad w_j \\ge 0, \\quad \\sum_j w_j = 1. $$",
            "",
            "Os pesos sao escolhidos para minimizar o erro quadratico medio no periodo pre-tratamento:",
            "",
            "$$ \\min_w \\sum_{t \\in T_0} (Y_{1t} - \\sum_j w_j Y_{jt})^2. $$",
            "",
            "O efeito estimado no pos-tratamento e:",
            "",
            "$$ \\hat{\\alpha}_{1t} = Y_{1t} - \\hat{Y}_{1t}(0). $$",
            "",
            "Todos os outcomes sao normalizados como indice com outubro de 2024 igual a 100 para ajuste dos pesos. Os efeitos finais sao reconvertidos para unidades originais.",
            "",
            "## 5. Testes de inferencia e robustez",
            "",
            "A inferencia segue praticas usuais de controle sintetico. Primeiro, roda-se placebo em espaco: cada municipio do pool pareado e tratado como se tivesse recebido a politica, e calcula-se sua razao RMSPE pos/pre. O p-valor empirico compara a razao de Bento com a distribuicao placebo. Segundo, roda-se um placebo temporal, com falso tratamento em maio de 2024 e falso pos-tratamento ate outubro de 2024. Terceiro, faz-se leave-one-out, retirando doadores com peso relevante para verificar se o resultado depende de uma unica unidade.",
            "",
            "## 6. Resultados",
            "",
            "A Tabela 1 resume os resultados principais.",
            "",
            "| Outcome | Pool | RMSPE pre | Razao pos/pre | p placebo | Efeito medio pos | Observado ultimo mes | Synth ultimo mes | Efeito ultimo mes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {OUTCOME_LABELS[row['outcome']]} | {row['pool']} | {fmt(row['pre_rmspe_index'])} | {fmt(row['post_pre_rmspe_ratio'])} | {fmt(row['p_value_space_placebo'], 3)} | {fmt(row['mean_post_effect_value'], 1)} | {fmt0(row['last_actual_value'])} | {fmt0(row['last_synth_value'])} | {fmt0(row['last_effect_value'])} |"
        )

    lines.extend(
        [
            "",
            "### 6.1 Bolsa Familia",
            "",
            f"No pool restrito ao Rio Grande do Sul, o controle sintetico projeta {fmt0(bf_rs['last_synth_value'])} familias beneficiarias em marco de 2026, contra {fmt0(bf_rs['last_actual_value'])} observadas. O efeito estimado e de {fmt0(bf_rs['last_effect_value'])} familias no ultimo mes e media de {fmt(bf_rs['mean_post_effect_value'], 1)} familias por mes no pos-tratamento. A razao RMSPE pos/pre e {fmt(bf_rs['post_pre_rmspe_ratio'])}, com p-valor empirico {fmt(bf_rs['p_value_space_placebo'], 3)}. O leave-one-out preserva efeitos negativos no ultimo mes, entre {fmt0(bf_rs['leave_one_out_last_effect_min'])} e {fmt0(bf_rs['leave_one_out_last_effect_max'])}.",
            "",
            "A evidencia e, portanto, consistente com queda adicional do numero de beneficiarios em Bento Goncalves apos novembro de 2024. A qualificacao importante e que o p-valor e moderado, nao definitivo, e a interpretacao causal depende da ausencia de choques locais simultaneos que afetem Bento diferentemente dos doadores.",
            "",
            "### 6.2 Emprego formal",
            "",
            f"Para estoque de empregos formais, o efeito estimado no pool RS e pequeno: {fmt0(stock_rs['last_effect_value'])} empregos no ultimo mes, com p-valor {fmt(stock_rs['p_value_space_placebo'], 3)}. Esse resultado nao sustenta, sozinho, uma leitura de aumento robusto de empregabilidade formal. Para admissoes, ha um resultado estatisticamente mais destacado no pool RS, mas o efeito em unidades e pequeno e o leave-one-out cruza sinais. Para desligamentos, o ultimo mes mostra aumento, mas a evidencia placebo e fraca.",
            "",
            "Assim, a leitura mais conservadora e que o efeito mais robusto aparece na saida do Bolsa Familia, enquanto os dados de emprego formal nao confirmam uma melhora causal forte de mercado de trabalho.",
            "",
            "## 7. Ameacas a identificacao",
            "",
            "Quatro ameacas merecem destaque. Primeiro, revisoes federais do Cadastro Unico podem gerar quedas no Bolsa Familia independentemente da politica municipal. Segundo, a calamidade no Rio Grande do Sul em 2024 pode afetar tanto mercado de trabalho quanto transferencias, embora o pool RS reduza parcialmente essa preocupacao. Terceiro, a politica combina componentes distintos: fiscalizacao cadastral, orientacao, encaminhamento a vagas e possiveis mudancas administrativas. O desenho estima um efeito reduzido da estrategia como pacote, nao de cada componente. Quarto, saida do beneficio nao implica necessariamente melhora de renda ou bem-estar.",
            "",
            "## 8. Conclusao",
            "",
            "O exercicio de controle sintetico sugere que Bento Goncalves teve queda adicional no numero de familias beneficiarias do Bolsa Familia apos novembro de 2024. A magnitude no pool RS e de aproximadamente 354 familias a menos em marco de 2026, com evidencia placebo moderada. Entretanto, os outcomes de emprego formal nao mostram efeito robusto comparavel. A conclusao substantiva e que ha evidencia sugestiva de efeito sobre permanencia no programa, mas ainda nao evidencia forte de que a politica tenha aumentado empregabilidade formal em magnitude suficiente para explicar toda a queda.",
            "",
            "## Referencias e fontes de dados",
            "",
            "- MDS/VISDATA. Endpoint publico de series municipais do Bolsa Familia: https://aplicacoes.mds.gov.br/sagi/servicos/misocial",
            "- Ministerio do Trabalho e Emprego. Novo CAGED, marco de 2026, Tabelas: https://www.gov.br/trabalho-e-emprego/pt-br/assuntos/estatisticas-trabalho/novo-caged/2026/marco/pagina-inicial",
            "- Abadie, A., Diamond, A., and Hainmueller, J. (2010). Synthetic Control Methods for Comparative Case Studies.",
            "- Abadie, A. (2021). Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects.",
            "",
            "## ApÃªndice: figuras",
            "",
            "![Bolsa Familia - RS](../outputs/figures/synth_bolsa_familia_rs.svg)",
            "",
            "![Estoque de emprego formal - RS](../outputs/figures/synth_emprego_estoque_rs.svg)",
        ]
    )
    return "\n".join(lines) + "\n"


class ArticlePdf:
    def __init__(self, path: Path) -> None:
        self.canvas = canvas.Canvas(str(path), pagesize=A4)
        self.width, self.height = A4
        self.margin = 48
        self.y = self.height - self.margin
        self.page = 1
        self.canvas.setTitle("Artigo econometrico - Bento Goncalves")

    def clean(self, text: object) -> str:
        return str(text).encode("latin-1", errors="replace").decode("latin-1")

    def ensure(self, space: float) -> None:
        if self.y - space < self.margin:
            self.footer()
            self.canvas.showPage()
            self.page += 1
            self.y = self.height - self.margin

    def footer(self) -> None:
        self.canvas.setFont("Times-Roman", 8)
        self.canvas.setFillColor(colors.HexColor("#555555"))
        self.canvas.drawCentredString(self.width / 2, 24, f"{self.page}")

    def line(self, text: str, size: int = 10, bold: bool = False, italic: bool = False) -> None:
        self.ensure(size + 7)
        if bold:
            font = "Times-Bold"
        elif italic:
            font = "Times-Italic"
        else:
            font = "Times-Roman"
        self.canvas.setFont(font, size)
        self.canvas.setFillColor(colors.black)
        self.canvas.drawString(self.margin, self.y, self.clean(text))
        self.y -= size + 5

    def wrap(self, text: str, max_chars: int = 96) -> list[str]:
        words = self.clean(text).split()
        rows: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if len(candidate) > max_chars and current:
                rows.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            rows.append(" ".join(current))
        return rows

    def paragraph(self, text: str, size: int = 10, indent: bool = True) -> None:
        first = True
        for row in self.wrap(text):
            self.ensure(size + 6)
            self.canvas.setFont("Times-Roman", size)
            x = self.margin + (18 if indent and first else 0)
            self.canvas.drawString(x, self.y, row)
            self.y -= size + 4
            first = False
        self.y -= 4

    def heading(self, text: str, level: int = 1) -> None:
        self.y -= 4
        self.line(text, size=13 if level == 1 else 11, bold=True)
        self.y -= 2

    def small_table(self, headers: list[str], rows: list[list[str]], title: str) -> None:
        self.ensure(24 + 16 * (len(rows) + 2))
        self.line(title, size=10, bold=True)
        col_widths = [154, 44, 46, 48, 44, 58, 58, 58, 58]
        x = self.margin
        self.canvas.setFont("Times-Bold", 7)
        self.canvas.setFillColor(colors.black)
        for header, w in zip(headers, col_widths):
            self.canvas.drawString(x, self.y, self.clean(header)[:22])
            x += w
        self.y -= 10
        self.canvas.setStrokeColor(colors.black)
        self.canvas.line(self.margin, self.y + 4, self.width - self.margin, self.y + 4)
        self.canvas.setFont("Times-Roman", 7)
        for row in rows:
            self.ensure(12)
            x = self.margin
            for value, w in zip(row, col_widths):
                self.canvas.drawString(x, self.y, self.clean(value)[:24])
                x += w
            self.y -= 10
        self.y -= 8

    def chart(self, title: str, df: pd.DataFrame) -> None:
        self.ensure(180)
        chart_w = self.width - 2 * self.margin
        chart_h = 116
        x0 = self.margin
        y0 = self.y - chart_h - 18
        self.line(title, size=10, bold=True)
        actual = df["actual_index"].astype(float).tolist()
        synth = df["synth_index"].astype(float).tolist()
        ymin = min(min(actual), min(synth)) - 4
        ymax = max(max(actual), max(synth)) + 4
        denom = max(ymax - ymin, 1e-9)

        def xy(idx: int, value: float) -> tuple[float, float]:
            x = x0 + idx * chart_w / max(len(df) - 1, 1)
            yy = y0 + (value - ymin) * chart_h / denom
            return x, yy

        self.canvas.setStrokeColor(colors.black)
        self.canvas.setLineWidth(0.7)
        self.canvas.line(x0, y0, x0 + chart_w, y0)
        self.canvas.line(x0, y0, x0, y0 + chart_h)
        matches = df.index[df["anomes"] == 202411].tolist()
        if matches:
            x, _ = xy(matches[0], ymin)
            self.canvas.setStrokeColor(colors.HexColor("#777777"))
            self.canvas.setDash(3, 3)
            self.canvas.line(x, y0, x, y0 + chart_h)
            self.canvas.setDash()
        for series, color in [(synth, colors.HexColor("#3578b8")), (actual, colors.HexColor("#d14b3f"))]:
            self.canvas.setStrokeColor(color)
            self.canvas.setLineWidth(1.3)
            for idx in range(len(series) - 1):
                self.canvas.line(*xy(idx, series[idx]), *xy(idx + 1, series[idx + 1]))
        self.canvas.setFont("Times-Roman", 8)
        self.canvas.setFillColor(colors.HexColor("#555555"))
        self.canvas.drawString(x0, y0 - 12, str(int(df.iloc[0]["anomes"])))
        self.canvas.drawRightString(x0 + chart_w, y0 - 12, str(int(df.iloc[-1]["anomes"])))
        self.canvas.setFillColor(colors.HexColor("#d14b3f"))
        self.canvas.drawString(x0 + chart_w - 130, y0 + chart_h + 2, "Bento")
        self.canvas.setFillColor(colors.HexColor("#3578b8"))
        self.canvas.drawString(x0 + chart_w - 80, y0 + chart_h + 2, "Synth")
        self.y = y0 - 28

    def save(self) -> None:
        self.footer()
        self.canvas.save()


def build_pdf(summary: pd.DataFrame, path: Path) -> None:
    pdf = ArticlePdf(path)
    pdf.canvas.setFont("Times-Bold", 15)
    title = "A politica municipal de Bento Goncalves e a queda no Bolsa Familia:"
    subtitle = "evidencia por controle sintetico"
    pdf.canvas.drawCentredString(pdf.width / 2, pdf.y, pdf.clean(title))
    pdf.y -= 18
    pdf.canvas.drawCentredString(pdf.width / 2, pdf.y, pdf.clean(subtitle))
    pdf.y -= 24
    pdf.canvas.setFont("Times-Roman", 10)
    pdf.canvas.drawCentredString(pdf.width / 2, pdf.y, "Versao gerada automaticamente")
    pdf.y -= 26

    pdf.line("Resumo", size=11, bold=True)
    pdf.paragraph(
        "Este artigo estima o efeito associado a uma estrategia municipal de busca ativa, verificacao cadastral e encaminhamento ao mercado de trabalho em Bento Goncalves (RS). O desenho empirico combina pareamento em caracteristicas observaveis e defasagens da variavel dependente com controle sintetico. A evidencia principal indica queda adicional de beneficiarios do Bolsa Familia; no pool restrito ao Rio Grande do Sul, Bento apresenta 1.296 familias beneficiarias em marco de 2026 contra 1.650 no contrafactual sintetico, uma diferenca de -354 familias. Para emprego formal, os efeitos sao pequenos ou instaveis.",
        indent=False,
    )
    pdf.line("Palavras-chave: Bolsa Familia; controle sintetico; municipios; mercado de trabalho.", size=9)
    pdf.line("JEL: C23; C54; H53; I38; J21.", size=9)

    sections = [
        (
            "1. Introducao",
            "A avaliacao de politicas locais de assistencia social enfrenta um problema classico de identificacao: municipios adotam intervencoes em momentos nao aleatorios e, ao mesmo tempo, estao expostos a choques nacionais, revisoes cadastrais federais e dinamicas locais de mercado de trabalho. A pergunta empirica deste artigo e se a queda observada em Bento Goncalves pode ser interpretada como efeito causal da politica municipal, ou se apenas reflete tendencias comuns a municipios comparaveis.",
        ),
        (
            "2. Dados",
            "O painel municipal mensal combina MDS/VISDATA para Bolsa Familia e a Tabela 8 do Novo CAGED/MTE para estoque, admissoes, desligamentos e saldo de emprego formal. A unidade tratada e Bento Goncalves, codigo MDS 430210. O periodo pre-tratamento vai de marco de 2023 a outubro de 2024; o periodo pos-tratamento vai de novembro de 2024 a marco de 2026.",
        ),
        (
            "3. Estrategia empirica",
            "A estrategia tem duas etapas. Primeiro, restringe-se o conjunto de doadores por pareamento em observaveis e trajetoria pre-tratamento. As covariadas incluem nivel medio pre, inclinacao pre, variacao percentual pre, volatilidade pre e defasagens mensais da propria variavel dependente. Segundo, estima-se uma combinacao convexa de municipios doadores que minimiza o erro quadratico medio no periodo pre-tratamento.",
        ),
        (
            "4. Inferencia",
            "A inferencia usa placebos em espaco, razao RMSPE pos/pre, placebo temporal com falso tratamento em maio de 2024 e leave-one-out dos doadores com peso relevante. O p-valor empirico compara a razao RMSPE pos/pre de Bento com a distribuicao obtida ao tratar cada doador como placebo.",
        ),
    ]
    for heading, text in sections:
        pdf.heading(heading)
        pdf.paragraph(text)

    headers = [
        "Outcome",
        "Pool",
        "RMSPE",
        "Pos/pre",
        "p",
        "Efeito medio",
        "Obs.",
        "Synth",
        "Efeito",
    ]
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            [
                OUTCOME_LABELS[row["outcome"]],
                row["pool"],
                fmt(row["pre_rmspe_index"]),
                fmt(row["post_pre_rmspe_ratio"]),
                fmt(row["p_value_space_placebo"], 3),
                fmt(row["mean_post_effect_value"], 1),
                fmt0(row["last_actual_value"]),
                fmt0(row["last_synth_value"]),
                fmt0(row["last_effect_value"]),
            ]
        )
    pdf.heading("5. Resultados")
    pdf.small_table(headers, rows, "Tabela 1. Resultados principais por outcome")

    bf_rs = summary[(summary["outcome"] == "bolsa_familia") & (summary["pool"] == "rs")].iloc[0]
    stock_rs = summary[(summary["outcome"] == "emprego_estoque") & (summary["pool"] == "rs")].iloc[0]
    pdf.paragraph(
        f"No pool restrito ao Rio Grande do Sul, o controle sintetico projeta {fmt0(bf_rs['last_synth_value'])} familias beneficiarias em marco de 2026, contra {fmt0(bf_rs['last_actual_value'])} observadas. O efeito estimado e de {fmt0(bf_rs['last_effect_value'])} familias no ultimo mes, com p-valor empirico {fmt(bf_rs['p_value_space_placebo'], 3)}. O leave-one-out preserva efeitos negativos, entre {fmt0(bf_rs['leave_one_out_last_effect_min'])} e {fmt0(bf_rs['leave_one_out_last_effect_max'])} familias."
    )
    pdf.paragraph(
        f"Para estoque de empregos formais, o efeito estimado no pool RS e pequeno: {fmt0(stock_rs['last_effect_value'])} empregos no ultimo mes, com p-valor {fmt(stock_rs['p_value_space_placebo'], 3)}. Os outcomes de admissoes e desligamentos tampouco sustentam uma interpretacao robusta de melhora de empregabilidade formal."
    )

    bf_ts = pd.read_csv(TABLES / "synth_bolsa_familia_rs_timeseries.csv")
    stock_ts = pd.read_csv(TABLES / "synth_emprego_estoque_rs_timeseries.csv")
    pdf.chart("Figura 1. Bolsa Familia: Bento vs controle sintetico (pool RS)", bf_ts)
    pdf.chart("Figura 2. Estoque formal: Bento vs controle sintetico (pool RS)", stock_ts)

    pdf.heading("6. Ameacas a identificacao")
    pdf.paragraph(
        "As principais ameacas sao revisoes federais do Cadastro Unico, choques locais simultaneos, os efeitos da calamidade no Rio Grande do Sul em 2024 e a composicao da propria politica, que combina encaminhamento ao trabalho e verificacao cadastral. Alem disso, saida do beneficio nao implica automaticamente melhora de renda ou bem-estar."
    )

    pdf.heading("7. Conclusao")
    pdf.paragraph(
        "O exercicio sugere queda adicional no numero de familias beneficiarias do Bolsa Familia em Bento Goncalves apos novembro de 2024. A magnitude no pool RS e de aproximadamente 354 familias a menos em marco de 2026, com evidencia placebo moderada. Entretanto, os outcomes de emprego formal nao mostram efeito robusto comparavel. A conclusao substantiva e que ha evidencia sugestiva de efeito sobre permanencia no programa, mas ainda nao evidencia forte de aumento de empregabilidade formal."
    )

    pdf.heading("Referencias e fontes")
    pdf.paragraph(
        "MDS/VISDATA, endpoint publico de series municipais do Bolsa Familia: https://aplicacoes.mds.gov.br/sagi/servicos/misocial",
        indent=False,
    )
    pdf.paragraph(
        "Ministerio do Trabalho e Emprego, Novo CAGED, marco de 2026: https://www.gov.br/trabalho-e-emprego/pt-br/assuntos/estatisticas-trabalho/novo-caged/2026/marco/pagina-inicial",
        indent=False,
    )
    pdf.paragraph(
        "Abadie, Diamond e Hainmueller (2010); Abadie (2021).",
        indent=False,
    )
    pdf.save()


def main() -> None:
    summary = pd.read_csv(TABLES / "synth_tests_summary.csv")
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = build_markdown(summary)
    (REPORTS / "econometric_article.md").write_text(md, encoding="utf-8")
    build_pdf(summary, REPORTS / "econometric_article.pdf")
    print(REPORTS / "econometric_article.pdf")
    print(REPORTS / "econometric_article.md")


if __name__ == "__main__":
    main()
