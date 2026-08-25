from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template

from .metrics import drawdown_series


PAGE_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Quantitative Portfolio Research Pipeline</title>
  <meta name="description" content="Walk-forward ETF portfolio optimization research by Nathan Gomes.">
  <link rel="icon" href="data:,">
  <style>
    :root{--ink:#151719;--muted:#6f7479;--line:#dfe2e1;--green:#176b55;--green-soft:#eaf2ef;--paper:#f7f8f7;--white:#fff}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,Arial,sans-serif;line-height:1.5}a{color:inherit}
    .shell{max-width:1180px;margin:0 auto;padding:0 32px 64px}.topbar{display:flex;justify-content:space-between;gap:24px;padding:24px 0;border-bottom:1px solid var(--line);font:12px ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase;letter-spacing:.08em}.topbar a{text-decoration:none;color:var(--muted)}
    .hero{padding:64px 0 44px}.eyebrow{color:var(--green);font:12px ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase;letter-spacing:.1em}.hero h1{max-width:850px;margin:16px 0 18px;font-size:clamp(38px,6vw,72px);line-height:1.02;letter-spacing:0}.hero p{max-width:760px;margin:0;color:#42474b;font-size:18px}.notice{display:inline-flex;gap:9px;align-items:center;margin-top:24px;padding:9px 12px;border:1px solid #b8d3c9;background:var(--green-soft);color:var(--green);font:11px ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase}.notice i{width:7px;height:7px;border-radius:50%;background:var(--green)}
    .metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border-top:1px solid var(--ink);border-bottom:1px solid var(--line);background:var(--white)}.metric{min-width:0;padding:22px 18px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric span{display:block;color:var(--muted);font:10px ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase}.metric strong{display:block;margin-top:9px;font-size:25px;font-weight:600}.metric small{color:var(--muted)}
    section{padding:48px 0;border-bottom:1px solid var(--line)}.section-head{display:flex;justify-content:space-between;gap:30px;align-items:end;margin-bottom:24px}.section-head h2{margin:0;font-size:28px}.section-head p{max-width:560px;margin:0;color:var(--muted);text-align:right}.chart{min-width:0;min-height:380px;background:var(--white);border:1px solid var(--line)}.grid{display:grid;grid-template-columns:1.25fr .75fr;gap:18px}.grid>*{min-width:0}.panel{min-width:0;background:var(--white);border:1px solid var(--line);padding:24px}.panel h3{margin:0 0 18px;font-size:18px}.weights{display:grid;gap:12px}.weight{display:grid;grid-template-columns:52px 1fr 52px;gap:12px;align-items:center;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}.bar{height:8px;background:#edf0ef}.bar i{display:block;height:100%;background:var(--green)}
    table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:11px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font:10px ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase}.method{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}.method div{padding:22px;background:var(--white)}.method span{color:var(--green);font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.method h3{margin:10px 0 7px;font-size:16px}.method p{margin:0;color:var(--muted);font-size:14px}.footer{padding-top:28px;color:var(--muted);font-size:13px}
    @media(max-width:760px){.shell{padding:0 18px 42px}.hero{padding:44px 0 34px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metric{border-bottom:1px solid var(--line)}.metric:nth-child(2n){border-right:0}.grid,.method{grid-template-columns:1fr}.section-head{display:block}.section-head p{margin-top:8px;text-align:left}.chart{min-height:320px}.topbar{align-items:flex-start;flex-direction:column}.metric strong{font-size:22px}.panel table{table-layout:fixed;font-size:9px}.panel th,.panel td{padding:8px 3px;overflow-wrap:anywhere}}
  </style>
</head>
<body><div class="shell">
  <header class="topbar"><strong>Nathan Gomes / Quant Research</strong><a href="Project-Quant-Portfolio.dc.html">Read the case study</a></header>
  <main>
    <div class="hero"><div class="eyebrow">Research-to-production portfolio analytics</div><h1>Quantitative Portfolio Research Pipeline</h1><p>A reproducible ETF research workflow using Python, pandas, SQL, constrained optimization, transaction-cost-aware walk-forward testing, and automated validation.</p><div class="notice"><i></i>Historical research only / no live trading</div></div>
    <div class="metrics">
      {% for metric in metrics %}<div class="metric"><span>{{ metric.label }}</span><strong>{{ metric.value }}</strong><small>{{ metric.detail }}</small></div>{% endfor %}
    </div>
    <section><div class="section-head"><h2>Out-of-sample performance</h2><p>Weights are estimated from the previous {{ config.lookback_months }} months, then held until the next quarterly rebalance. Costs are charged when allocations change.</p></div><div class="chart">{{ growth_chart }}</div></section>
    <section><div class="section-head"><h2>Risk and allocation</h2><p>The strategy caps every asset at {{ max_weight }} and remains fully invested without leverage or short positions.</p></div><div class="grid"><div class="chart">{{ drawdown_chart }}</div><div class="panel"><h3>Latest optimized weights</h3><div class="weights">{% for row in weights %}<div class="weight"><b>{{ row.ticker }}</b><div class="bar"><i style="width:{{ row.bar }}%"></i></div><span>{{ row.value }}</span></div>{% endfor %}</div></div></section>
    <section><div class="section-head"><h2>Cross-asset structure</h2><p>Daily return correlations are estimated from the frozen research dataset.</p></div><div class="grid"><div class="chart">{{ correlation_chart }}</div><div class="panel"><h3>Strategy comparison</h3>{{ comparison_table }}</div></div></section>
    <section><div class="section-head"><h2>Research controls</h2><p>The configuration file controls the experiment, so changing an assumption reruns the complete pipeline and report.</p></div><div class="method">
      <div><span>01</span><h3>Frozen inputs</h3><p>Six ETF histories from {{ config.data_start }} through {{ config.data_end }} are stored locally and loaded into SQLite.</p></div>
      <div><span>02</span><h3>No look-ahead</h3><p>Each rebalance uses only observations available on or before that date.</p></div>
      <div><span>03</span><h3>Trading friction</h3><p>{{ config.transaction_cost_bps }} basis points are charged against portfolio turnover at every rebalance.</p></div>
      <div><span>04</span><h3>Reproducible checks</h3><p>Pytest verifies returns, weights, drawdowns, constraints, and backtest chronology.</p></div>
    </div></section>
  </main><footer class="footer">Generated {{ generated_at }} from a frozen historical dataset. Results are educational and are not investment advice.</footer>
</div></body></html>"""
)


def _chart_html(figure: go.Figure) -> str:
    return pio.to_html(figure, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False, "responsive": True})


def _base_layout(figure: go.Figure, title: str) -> None:
    figure.update_layout(
        title={"text": title, "font": {"size": 15}},
        margin=dict(l=48, r=24, t=54, b=42),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, Arial, sans-serif", "color": "#34383b", "size": 12},
        legend={"orientation": "h", "y": 1.08},
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False, linecolor="#dfe2e1")
    figure.update_yaxes(gridcolor="#edf0ef", zerolinecolor="#dfe2e1")


def generate_report(
    output_path: Path,
    config: dict,
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    correlation: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    growth = (1.0 + returns).cumprod()
    growth_figure = go.Figure()
    colors = ["#176b55", "#7c858b"]
    for color, column in zip(colors, growth.columns):
        growth_figure.add_trace(go.Scatter(x=growth.index, y=growth[column], name=column, line={"color": color, "width": 2.4}))
    _base_layout(growth_figure, "Growth of $1")
    growth_figure.update_yaxes(tickprefix="$", tickformat=".2f")

    drawdown_figure = go.Figure()
    for color, column in zip(colors, returns.columns):
        drawdown_figure.add_trace(go.Scatter(x=returns.index, y=drawdown_series(returns[column]), name=column, line={"color": color}))
    _base_layout(drawdown_figure, "Drawdown")
    drawdown_figure.update_yaxes(tickformat=".0%")

    correlation_figure = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            zmin=-1,
            zmax=1,
            colorscale=[[0, "#9daca7"], [0.5, "#f7f8f7"], [1, "#176b55"]],
            text=correlation.round(2).astype(str).values,
            texttemplate="%{text}",
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        )
    )
    _base_layout(correlation_figure, "Daily return correlation")

    strategy_name = returns.columns[0]
    summary = summaries.loc[strategy_name]
    latest = weights.iloc[-1].sort_values(ascending=False)
    metrics = [
        {"label": "Annual return", "value": f"{summary['annual_return']:.1%}", "detail": "geometric"},
        {"label": "Annual volatility", "value": f"{summary['annual_volatility']:.1%}", "detail": "daily observations"},
        {"label": "Sharpe ratio", "value": f"{summary['sharpe_ratio']:.2f}", "detail": f"{config['risk_free_rate']:.0%} risk-free rate"},
        {"label": "Maximum drawdown", "value": f"{summary['maximum_drawdown']:.1%}", "detail": "peak to trough"},
        {"label": "Rebalances", "value": str(len(weights)), "detail": "walk-forward"},
    ]
    weight_rows = [{"ticker": ticker, "bar": round(value * 100 / config["maximum_asset_weight"], 1), "value": f"{value:.1%}"} for ticker, value in latest.items()]
    comparison = summaries.rename_axis("Strategy").reset_index().copy()
    for column in ["annual_return", "annual_volatility", "maximum_drawdown", "daily_var_95"]:
        comparison[column] = comparison[column].map(lambda value: f"{value:.1%}")
    comparison["sharpe_ratio"] = comparison["sharpe_ratio"].map(lambda value: f"{value:.2f}")
    comparison.columns = ["Strategy", "Return", "Volatility", "Sharpe", "Max drawdown", "Daily VaR"]

    html = PAGE_TEMPLATE.render(
        config=config,
        max_weight=f"{config['maximum_asset_weight']:.0%}",
        metrics=metrics,
        weights=weight_rows,
        growth_chart=_chart_html(growth_figure),
        drawdown_chart=_chart_html(drawdown_figure),
        correlation_chart=_chart_html(correlation_figure),
        comparison_table=comparison.to_html(index=False, border=0),
        generated_at=pd.Timestamp.now().strftime("%B %d, %Y"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"metrics": metrics, "latest_weights": latest.to_dict()}, indent=2), encoding="utf-8")
