from __future__ import annotations

from pathlib import Path

import pandas as pd


def _points(values: list[float], width: int, height: int, pad: int, ymin: float, ymax: float) -> str:
    denom = max(ymax - ymin, 1e-9)
    step = (width - 2 * pad) / max(len(values) - 1, 1)
    coords = []
    for i, value in enumerate(values):
        x = pad + i * step
        y = height - pad - ((value - ymin) / denom) * (height - 2 * pad)
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def write_svg_timeseries(
    df: pd.DataFrame,
    path: Path,
    title: str,
    intervention_month: int,
    y_axis_note: str = "Indice: outubro de 2024 = 100",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, pad = 880, 420, 52
    actual = df["actual_index"].tolist()
    synth = df["synth_index"].tolist()
    ymin = min(min(actual), min(synth)) - 5
    ymax = max(max(actual), max(synth)) + 5
    intervention_idx = df.index[df["anomes"] == intervention_month]
    x_intervention = None
    if len(intervention_idx):
        step = (width - 2 * pad) / max(len(df) - 1, 1)
        x_intervention = pad + int(intervention_idx[0]) * step
    actual_points = _points(actual, width, height, pad, ymin, ymax)
    synth_points = _points(synth, width, height, pad, ymin, ymax)
    x_labels = []
    for idx in [0, len(df) // 2, len(df) - 1]:
        month = int(df.iloc[idx]["anomes"])
        x = pad + idx * ((width - 2 * pad) / max(len(df) - 1, 1))
        x_labels.append(f'<text x="{x:.1f}" y="{height - 16}" text-anchor="middle">{month}</text>')
    line_intervention = ""
    if x_intervention is not None:
        line_intervention = (
            f'<line x1="{x_intervention:.1f}" y1="{pad}" x2="{x_intervention:.1f}" '
            f'y2="{height - pad}" stroke="#777" stroke-dasharray="5 5"/>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{pad}" y="28" font-family="Arial" font-size="18" font-weight="700">{title}</text>
  <text x="{pad}" y="48" font-family="Arial" font-size="12" fill="#555">{y_axis_note}</text>
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>
  {line_intervention}
  <polyline points="{synth_points}" fill="none" stroke="#3578b8" stroke-width="3"/>
  <polyline points="{actual_points}" fill="none" stroke="#d14b3f" stroke-width="3"/>
  <text x="{width - 210}" y="38" font-family="Arial" font-size="13" fill="#d14b3f">Bento Goncalves</text>
  <text x="{width - 210}" y="58" font-family="Arial" font-size="13" fill="#3578b8">Controle sintetico</text>
  <text x="14" y="{pad+4}" font-family="Arial" font-size="12" fill="#555">{ymax:.0f}</text>
  <text x="14" y="{height-pad+4}" font-family="Arial" font-size="12" fill="#555">{ymin:.0f}</text>
  {"".join(x_labels)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_report(
    path: Path,
    national: dict,
    rs: dict,
    national_placebos: pd.DataFrame,
    rs_placebos: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def block(label: str, result: dict, placebos: pd.DataFrame) -> str:
        m = result["metrics"]
        p_value = None
        if not placebos.empty:
            p_value = (1 + (placebos["post_pre_rmspe_ratio"] >= m["post_pre_rmspe_ratio"]).sum()) / (
                1 + len(placebos)
            )
        p_text = "n/a" if p_value is None else f"{p_value:.3f}"
        top_weights = (
            result["weights"]
            .sort_values("weight", ascending=False)
            .head(8)
            .merge(result["donor_labels"], on="id_municipio_6", how="left")
        )
        weights_md = "\n".join(
            f"- {row.get('municipio', row['id_municipio_6'])} ({row.get('sigla_uf', '')}): {row['weight']:.3f}"
            for _, row in top_weights.iterrows()
        )
        return f"""## {label}

- RMSPE pre: {m['pre_rmspe_index']:.2f} pontos de indice.
- Razao RMSPE pos/pre: {m['post_pre_rmspe_ratio']:.2f}; p-valor placebo aproximado: {p_text}.
- Efeito medio pos: {m['mean_post_effect_index']:.1f} pontos de indice, ou {m['mean_post_effect_families']:.0f} familias por mes.
- Ultimo mes: observado {m['last_actual_families']:.0f} familias vs contrafactual {m['last_synth_families']:.0f}; efeito {m['last_effect_families']:.0f} familias.
- Efeito acumulado no pos: {m['cumulative_effect_families_month']:.0f} familias-mes.

Principais pesos:

{weights_md}
"""

    text = f"""# Resultados iniciais

Analise gerada automaticamente para o outcome `familias beneficiarias do Bolsa Familia`.

Tratamento operacional: novembro de 2024. Pre-tratamento: marco de 2023 a outubro de 2024. Pos-tratamento: novembro de 2024 ao ultimo mes baixado.

Nota metodologica: o matching agora usa defasagens mensais de `y` normalizadas por outubro de 2024; o synth tambem usa toda a trajetoria pre-tratamento de `y` como matriz de preditores para os pesos.

{block("Pool Brasil", national, national_placebos)}

{block("Pool Rio Grande do Sul", rs, rs_placebos)}

## Leitura substantiva

A evidencia inicial favorece a hipotese de queda adicional de beneficiarios em Bento Goncalves depois de novembro de 2024, acima do que seria esperado por municipios parecidos na pre-trajetoria do Bolsa Familia. Ainda assim, este resultado mede saida do programa, nao necessariamente melhora de bem-estar.

## Pendencias

- Rodar CAGED municipal mensal para saldo/admissoes/desligamentos e testar empregabilidade.
- Adicionar PIB, populacao e despesas municipais via Base dos Dados assim que houver autenticacao Google Cloud local.
- Testar robustez com janelas alternativas, especialmente se a data correta do inicio da politica for revisada.
"""
    path.write_text(text, encoding="utf-8")


def write_pdf_report(
    path: Path,
    national: dict,
    rs: dict,
    national_placebos: pd.DataFrame,
    rs_placebos: pd.DataFrame,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin = 44
    y = height - margin

    def clean(text: object) -> str:
        return str(text).encode("latin-1", errors="replace").decode("latin-1")

    def new_page() -> None:
        nonlocal y
        c.showPage()
        y = height - margin

    def ensure(space: float) -> None:
        if y - space < margin:
            new_page()

    def line(text: str, size: int = 10, bold: bool = False, color=colors.black) -> None:
        nonlocal y
        ensure(size + 8)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.setFillColor(color)
        c.drawString(margin, y, clean(text))
        y -= size + 6

    def wrap(text: str, max_chars: int = 94) -> list[str]:
        words = clean(text).split()
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

    def paragraph(text: str, size: int = 10) -> None:
        nonlocal y
        for row in wrap(text):
            line(row, size=size)
        y -= 4

    def bullet(text: str) -> None:
        nonlocal y
        for idx, row in enumerate(wrap(text, max_chars=88)):
            prefix = "- " if idx == 0 else "  "
            line(prefix + row, size=9)
        y -= 1

    def p_value(metrics: dict, placebos: pd.DataFrame) -> str:
        if placebos.empty:
            return "n/a"
        p = (1 + (placebos["post_pre_rmspe_ratio"] >= metrics["post_pre_rmspe_ratio"]).sum()) / (
            1 + len(placebos)
        )
        return f"{p:.3f}"

    def pool_section(label: str, result: dict, placebos: pd.DataFrame) -> None:
        nonlocal y
        m = result["metrics"]
        line(label, size=14, bold=True, color=colors.HexColor("#1f3b5b"))
        bullet(f"RMSPE pre: {m['pre_rmspe_index']:.2f} pontos de indice.")
        bullet(
            f"Razao RMSPE pos/pre: {m['post_pre_rmspe_ratio']:.2f}; "
            f"p-valor placebo aproximado: {p_value(m, placebos)}."
        )
        bullet(
            f"Efeito medio pos: {m['mean_post_effect_index']:.1f} pontos de indice, "
            f"ou {m['mean_post_effect_families']:.0f} familias por mes."
        )
        bullet(
            f"Ultimo mes: observado {m['last_actual_families']:.0f} familias vs "
            f"contrafactual {m['last_synth_families']:.0f}; "
            f"efeito {m['last_effect_families']:.0f} familias."
        )
        weights = (
            result["weights"]
            .sort_values("weight", ascending=False)
            .head(6)
            .merge(result["donor_labels"], on="id_municipio_6", how="left")
        )
        line("Principais pesos:", size=10, bold=True)
        for _, row in weights.iterrows():
            bullet(f"{row.get('municipio', row['id_municipio_6'])} ({row.get('sigla_uf', '')}): {row['weight']:.3f}")
        y -= 6

    def chart(title: str, df: pd.DataFrame) -> None:
        nonlocal y
        ensure(210)
        chart_h = 150
        chart_w = width - 2 * margin
        x0 = margin
        y0 = y - chart_h - 24
        c.setFont("Helvetica-Bold", 11)
        c.setFillColor(colors.black)
        c.drawString(x0, y, clean(title))
        y -= 18
        actual = df["actual_index"].astype(float).tolist()
        synth = df["synth_index"].astype(float).tolist()
        ymin = min(min(actual), min(synth)) - 5
        ymax = max(max(actual), max(synth)) + 5
        denom = max(ymax - ymin, 1e-9)

        def xy(idx: int, value: float) -> tuple[float, float]:
            x = x0 + idx * chart_w / max(len(df) - 1, 1)
            yy = y0 + (value - ymin) * chart_h / denom
            return x, yy

        c.setStrokeColor(colors.HexColor("#444444"))
        c.line(x0, y0, x0 + chart_w, y0)
        c.line(x0, y0, x0, y0 + chart_h)
        for month in [202411]:
            matches = df.index[df["anomes"] == month].tolist()
            if matches:
                x, _ = xy(matches[0], ymin)
                c.setStrokeColor(colors.HexColor("#777777"))
                c.setDash(4, 4)
                c.line(x, y0, x, y0 + chart_h)
                c.setDash()
        for series, color in [
            (synth, colors.HexColor("#3578b8")),
            (actual, colors.HexColor("#d14b3f")),
        ]:
            c.setStrokeColor(color)
            c.setLineWidth(1.6)
            for idx in range(len(series) - 1):
                x1, y1 = xy(idx, series[idx])
                x2, y2 = xy(idx + 1, series[idx + 1])
                c.line(x1, y1, x2, y2)
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(x0, y0 - 14, str(int(df.iloc[0]["anomes"])))
        c.drawCentredString(x0 + chart_w / 2, y0 - 14, str(int(df.iloc[len(df) // 2]["anomes"])))
        c.drawRightString(x0 + chart_w, y0 - 14, str(int(df.iloc[-1]["anomes"])))
        c.setFillColor(colors.HexColor("#d14b3f"))
        c.drawString(x0 + chart_w - 160, y0 + chart_h + 2, "Bento Goncalves")
        c.setFillColor(colors.HexColor("#3578b8"))
        c.drawString(x0 + chart_w - 55, y0 + chart_h + 2, "Synth")
        y = y0 - 30

    c.setTitle("Resultados iniciais - Bento Goncalves Bolsa Familia")
    line("Bento Goncalves: Bolsa Familia e controle sintetico", size=16, bold=True)
    paragraph(
        "Outcome: familias beneficiarias do Bolsa Familia. Tratamento operacional: novembro "
        "de 2024. Pre-tratamento: marco de 2023 a outubro de 2024. Pos-tratamento: "
        "novembro de 2024 a marco de 2026."
    )
    paragraph(
        "Especificacao: matching em caracteristicas observaveis e defasagens mensais de y "
        "normalizadas por outubro de 2024; em seguida, synth com toda a trajetoria "
        "pre-tratamento de y como preditores dos pesos."
    )
    pool_section("Pool Brasil", national, national_placebos)
    pool_section("Pool Rio Grande do Sul", rs, rs_placebos)
    chart("Indice de beneficiarios - Pool Brasil", national["timeseries"])
    chart("Indice de beneficiarios - Pool Rio Grande do Sul", rs["timeseries"])
    line("Leitura substantiva", size=14, bold=True, color=colors.HexColor("#1f3b5b"))
    paragraph(
        "A evidencia inicial sugere queda adicional de beneficiarios em Bento Goncalves depois "
        "de novembro de 2024, especialmente no pool restrito ao RS. O resultado mede saida do "
        "programa, nao necessariamente melhora de bem-estar ou empregabilidade."
    )
    line("Pendencias", size=14, bold=True, color=colors.HexColor("#1f3b5b"))
    bullet("Rodar CAGED municipal mensal para saldo, admissoes e desligamentos.")
    bullet("Adicionar PIB, populacao e despesas municipais via Base dos Dados quando o Google Cloud local estiver autenticado.")
    bullet("Testar robustez com janelas alternativas de tratamento.")
    c.save()


def synth_p_value(metrics: dict, placebos: pd.DataFrame) -> float | None:
    if placebos.empty:
        return None
    return (1 + (placebos["post_pre_rmspe_ratio"] >= metrics["post_pre_rmspe_ratio"]).sum()) / (
        1 + len(placebos)
    )


def write_multi_outcome_markdown(path: Path, analyses: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Resultados com outcomes adicionais",
        "",
        "Especificacao: matching por caracteristicas pre e lags mensais de `y`, seguido de controle sintetico com toda a trajetoria pre-tratamento. Testes: placebos em espaco, razao MSPE pos/pre, placebo temporal em maio/2024 e leave-one-out dos doadores com peso relevante.",
        "",
    ]
    for analysis in analyses:
        rows.extend([f"## {analysis['label']}", ""])
        for pool_label, pool in analysis["pools"].items():
            metrics = pool["result"]["metrics"]
            p_value = synth_p_value(metrics, pool["placebos"])
            p_text = "n/a" if p_value is None else f"{p_value:.3f}"
            fake = pool["in_time"]["metrics"]
            loo = pool["leave_one_out"]
            loo_text = "n/a"
            if not loo.empty:
                loo_text = (
                    f"{loo['last_effect_value'].min():.0f} a "
                    f"{loo['last_effect_value'].max():.0f}"
                )
            rows.extend(
                [
                    f"### {pool_label}",
                    "",
                    f"- RMSPE pre: {metrics['pre_rmspe_index']:.2f}; razao pos/pre: {metrics['post_pre_rmspe_ratio']:.2f}; p placebo: {p_text}.",
                    f"- Efeito medio pos: {metrics['mean_post_effect_value']:.1f}; ultimo mes: observado {metrics['last_actual_value']:.0f}, synth {metrics['last_synth_value']:.0f}, efeito {metrics['last_effect_value']:.0f}.",
                    f"- Placebo temporal maio-outubro/2024: efeito medio {fake['mean_post_effect_index']:.2f} pontos de indice.",
                    f"- Leave-one-out, efeito no ultimo mes: {loo_text}.",
                    "",
                ]
            )
    path.write_text("\n".join(rows), encoding="utf-8")


def write_multi_outcome_pdf(path: Path, analyses: list[dict]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    margin = 42
    y = height - margin

    def clean(text: object) -> str:
        return str(text).encode("latin-1", errors="replace").decode("latin-1")

    def ensure(space: float) -> None:
        nonlocal y
        if y - space < margin:
            c.showPage()
            y = height - margin

    def text_line(text: str, size: int = 9, bold: bool = False, color=colors.black) -> None:
        nonlocal y
        ensure(size + 7)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(margin, y, clean(text))
        y -= size + 5

    def wrap(text: str, max_chars: int = 98) -> list[str]:
        words = clean(text).split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if len(candidate) > max_chars and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return lines

    def paragraph(text: str, size: int = 9) -> None:
        nonlocal y
        for row in wrap(text):
            text_line(row, size=size)
        y -= 3

    def bullet(text: str) -> None:
        for i, row in enumerate(wrap(text, 91)):
            text_line(("- " if i == 0 else "  ") + row, size=8)

    def chart(title: str, df: pd.DataFrame) -> None:
        nonlocal y
        ensure(128)
        chart_h = 82
        chart_w = width - 2 * margin
        x0 = margin
        y_top = y
        y0 = y_top - chart_h - 12
        text_line(title, size=8, bold=True)
        actual = df["actual_index"].astype(float).tolist()
        synth = df["synth_index"].astype(float).tolist()
        ymin = min(min(actual), min(synth)) - 4
        ymax = max(max(actual), max(synth)) + 4
        denom = max(ymax - ymin, 1e-9)

        def xy(idx: int, value: float) -> tuple[float, float]:
            x = x0 + idx * chart_w / max(len(df) - 1, 1)
            yy = y0 + (value - ymin) * chart_h / denom
            return x, yy

        c.setStrokeColor(colors.HexColor("#444444"))
        c.setLineWidth(0.7)
        c.line(x0, y0, x0 + chart_w, y0)
        c.line(x0, y0, x0, y0 + chart_h)
        for month in [202411]:
            matches = df.index[df["anomes"] == month].tolist()
            if matches:
                x, _ = xy(matches[0], ymin)
                c.setStrokeColor(colors.HexColor("#777777"))
                c.setDash(3, 3)
                c.line(x, y0, x, y0 + chart_h)
                c.setDash()
        for series, color in [
            (synth, colors.HexColor("#3578b8")),
            (actual, colors.HexColor("#d14b3f")),
        ]:
            c.setStrokeColor(color)
            c.setLineWidth(1.2)
            for idx in range(len(series) - 1):
                c.line(*xy(idx, series[idx]), *xy(idx + 1, series[idx + 1]))
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawString(x0, y0 - 10, str(int(df.iloc[0]["anomes"])))
        c.drawRightString(x0 + chart_w, y0 - 10, str(int(df.iloc[-1]["anomes"])))
        y = y0 - 18

    c.setTitle("Controle sintetico - Bento Goncalves")
    text_line("Bento Goncalves: controle sintetico", size=16, bold=True)
    paragraph(
        "Resultados para Bolsa Familia e emprego formal. Pre-tratamento: marco/2023 a outubro/2024; tratamento operacional: novembro/2024; pos: novembro/2024 a marco/2026."
    )
    paragraph(
        "Testes tipicos: qualidade do fit pre, placebos em espaco, razao MSPE pos/pre, placebo temporal com falso tratamento em maio/2024 e leave-one-out dos principais doadores."
    )

    for analysis in analyses:
        ensure(190)
        text_line(analysis["label"], size=13, bold=True, color=colors.HexColor("#1f3b5b"))
        for pool_label, pool in analysis["pools"].items():
            metrics = pool["result"]["metrics"]
            p_value = synth_p_value(metrics, pool["placebos"])
            p_text = "n/a" if p_value is None else f"{p_value:.3f}"
            fake = pool["in_time"]["metrics"]
            loo = pool["leave_one_out"]
            if loo.empty:
                loo_text = "n/a"
            else:
                loo_text = f"{loo['last_effect_value'].min():.0f} a {loo['last_effect_value'].max():.0f}"
            text_line(pool_label, size=10, bold=True)
            bullet(
                f"RMSPE pre {metrics['pre_rmspe_index']:.2f}; razao pos/pre "
                f"{metrics['post_pre_rmspe_ratio']:.2f}; p placebo {p_text}."
            )
            bullet(
                f"Ultimo mes: observado {metrics['last_actual_value']:.0f}, synth "
                f"{metrics['last_synth_value']:.0f}, efeito {metrics['last_effect_value']:.0f}."
            )
            bullet(
                f"Placebo temporal: efeito medio {fake['mean_post_effect_index']:.2f} p.p. "
                f"Leave-one-out no ultimo mes: {loo_text}."
            )
        chart(f"{analysis['label']} - Pool RS", analysis["pools"]["rs"]["result"]["timeseries"])
    c.save()
