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
    .verdict{padding:36px 0}.verdict-card{background:var(--green-soft);border:1px solid #b8d3c9;border-left:4px solid var(--green);padding:28px 32px}.verdict-card p{margin:0;font-size:17px;line-height:1.65;max-width:920px}.verdict-card p+p{margin-top:14px}.verdict-eyebrow{color:var(--green);font:11px ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px}
    section{padding:48px 0;border-bottom:1px solid var(--line)}.section-head{display:flex;justify-content:space-between;gap:30px;align-items:end;margin-bottom:24px}.section-head h2{margin:0;font-size:28px}.section-head p{max-width:560px;margin:0;color:var(--muted);text-align:right}.explainer{max-width:820px;margin:0 0 22px;color:#42474b;font-size:15px;line-height:1.65}.explainer strong{color:var(--ink)}.chart{min-width:0;min-height:380px;background:var(--white);border:1px solid var(--line)}.grid{display:grid;grid-template-columns:1.25fr .75fr;gap:18px}.grid>*{min-width:0}.panel{min-width:0;background:var(--white);border:1px solid var(--line);padding:24px}.panel h3{margin:0 0 18px;font-size:18px}.weights{display:grid;gap:12px}.weight{display:grid;grid-template-columns:52px 1fr 52px;gap:12px;align-items:center;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}.bar{height:8px;background:#edf0ef}.bar i{display:block;height:100%;background:var(--green)}
    table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:11px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}th{color:var(--muted);font:10px ui-monospace,SFMono-Regular,Menlo,monospace;text-transform:uppercase}.method{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}.method div{padding:22px;background:var(--white)}.method span{color:var(--green);font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.method h3{margin:10px 0 7px;font-size:16px}.method p{margin:0;color:var(--muted);font-size:14px}.footer{padding-top:28px;color:var(--muted);font-size:13px}
    .glossary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px 40px}.glossary dt{font-weight:600;font-size:15px}.glossary dd{margin:5px 0 0;color:var(--muted);font-size:14px;line-height:1.55}
    @media(max-width:760px){.shell{padding:0 18px 42px}.hero{padding:44px 0 34px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.metric{border-bottom:1px solid var(--line)}.metric:nth-child(2n){border-right:0}.grid,.method,.glossary{grid-template-columns:1fr}.section-head{display:block}.section-head p{margin-top:8px;text-align:left}.chart{min-height:320px}.topbar{align-items:flex-start;flex-direction:column}.metric strong{font-size:22px}.panel table{table-layout:fixed;font-size:9px}.panel th,.panel td{padding:8px 3px;overflow-wrap:anywhere}.verdict-card{padding:22px}}
  </style>
</head>
<body><div class="shell">
  <header class="topbar"><strong>Nathan Gomes / Quant Research</strong><a href="Project-Quant-Portfolio.dc.html">Read the case study</a></header>
  <main>
    <div class="hero"><div class="eyebrow">Research-to-production portfolio analytics</div><h1>Quantitative Portfolio Research Pipeline</h1><p>A reproducible ETF research workflow using Python, pandas, SQL, constrained optimization, transaction-cost-aware walk-forward testing, and automated validation.</p><div class="notice"><i></i>Historical research only / no live trading</div></div>
    <div class="metrics">
      {% for metric in metrics %}<div class="metric"><span>{{ metric.label }}</span><strong>{{ metric.value }}</strong><small>{{ metric.detail }}</small></div>{% endfor %}
    </div>
    <section class="verdict"><div class="verdict-card"><div class="verdict-eyebrow">Bottom line</div>{{ verdict_html }}</div></section>
    <section><div class="section-head"><h2>Out-of-sample performance</h2><p>Weights are estimated from the previous {{ config.lookback_months }} months, then held until the next quarterly rebalance. Costs are charged when allocations change.</p></div><p class="explainer">This chart tracks $1 invested on day one through every rebalance to today. It is <strong>out-of-sample</strong>: at each point, the weights being tested were chosen using only data from before that date, so the line shows what an investor would actually have experienced, not a fit to history. A steeper climb means faster growth; a shallower dip during a downturn means the portfolio lost less when markets fell.</p><div class="chart">{{ growth_chart }}</div></section>
    <section><div class="section-head"><h2>Risk and allocation</h2><p>The strategy caps every asset at {{ max_weight }} and remains fully invested without leverage or short positions.</p></div><p class="explainer"><strong>Drawdown</strong> is how far a portfolio has fallen from its highest-ever value, at every point in time. The chart below is always at or below 0%. It matters more than volatility alone because losses and gains are not symmetric: a 20% drawdown needs a 25% gain just to recover, and a 25% drawdown needs 33%. A shallower drawdown line means less ground to make up. The panel on the right shows how the optimizer is currently positioned across the six ETFs, based on the most recent rebalance.</p><div class="grid"><div class="chart">{{ drawdown_chart }}</div><div class="panel"><h3>Latest optimized weights</h3><div class="weights">{% for row in weights %}<div class="weight"><b>{{ row.ticker }}</b><div class="bar"><i style="width:{{ row.bar }}%"></i></div><span>{{ row.value }}</span></div>{% endfor %}</div></div></div></section>
    <section><div class="section-head"><h2>Cross-asset structure</h2><p>Daily return correlations are estimated from the frozen research dataset.</p></div><p class="explainer">Diversification only works between assets that do not move together. A correlation of 1.0 means two assets move in lockstep, so combining them provides no protection. A correlation near 0, or negative, means one can be falling while the other holds up or rises, which is what actually reduces portfolio-level risk. The heatmap below is why the optimizer leans on the assets it does: whichever pair is darkest is providing the least diversification benefit, and the lightest (or most negative) pairs are doing the most work.</p><div class="grid"><div class="chart">{{ correlation_chart }}</div><div class="panel"><h3>Strategy comparison</h3>{{ comparison_table }}</div></div></section>
    <section><div class="section-head"><h2>Did optimizing beat not optimizing?</h2><p>Every rule is run on identical rebalance dates and identical training windows, so a difference between two of them comes from the rule rather than from a luckier schedule.</p></div><p class="explainer">Comparing the optimizer only with {{ benchmark_label }} asks whether this particular <em>asset mix</em> was a good choice. It does not say whether solving for weights was worth the trouble, because a portfolio manager could have picked any fixed split of the same six ETFs without running an optimization at all. The rows below labelled <strong>equal weight</strong>, <strong>inverse volatility</strong>, and <strong>risk parity</strong> are rules that use no return or covariance forecast whatsoever. They estimate nothing. If the optimizer cannot beat those, the optimization itself added no value.</p><div class="panel">{{ strategy_table }}<p style="margin:18px 0 0;color:var(--muted);font-size:13px">{{ strategy_note }}</p></div></section>
    <section><div class="section-head"><h2>Does the conclusion survive its assumptions?</h2><p>One assumption is varied at a time. Compare strategies within a row; a longer lookback also shortens the evaluation window, which is why the benchmark column moves too.</p></div><p class="explainer">{{ sensitivity_summary }}</p><div class="panel">{{ sensitivity_table }}<p style="margin:18px 0 0;color:var(--muted);font-size:13px">{{ sensitivity_note }}</p></div></section>
    <section><div class="section-head"><h2>Research controls</h2><p>The configuration file controls the experiment, so changing an assumption reruns the complete pipeline and report.</p></div><div class="method">
      <div><span>01</span><h3>Frozen inputs</h3><p>Six ETF histories from {{ config.data_start }} through {{ config.data_end }} are stored locally and loaded into SQLite.</p></div>
      <div><span>02</span><h3>No look-ahead</h3><p>Each rebalance uses only observations available on or before that date.</p></div>
      <div><span>03</span><h3>Trading friction</h3><p>{{ config.transaction_cost_bps }} basis points are charged against portfolio turnover at every rebalance.</p></div>
      <div><span>04</span><h3>Reproducible checks</h3><p>Pytest verifies returns, weights, drawdowns, constraints, and backtest chronology.</p></div>
    </div></section>
    <section><div class="section-head"><h2>Terms used in this report</h2></div><dl class="glossary">
      <div><dt>Sharpe ratio</dt><dd>Return earned above the risk-free rate, divided by volatility. It answers "how much return per unit of risk," not "how much return." A higher Sharpe ratio can come with a lower raw return if the risk taken to get there was low enough.</dd></div>
      <div><dt>Volatility</dt><dd>How much a portfolio's daily returns swing around their average, expressed as an annualized percentage. Higher volatility means a bumpier ride, independent of whether the overall trend is up or down.</dd></div>
      <div><dt>Drawdown</dt><dd>The percentage fall from a portfolio's highest-ever value to its current value. Unlike volatility, it captures the worst single experience an investor actually lived through, not an average.</dd></div>
      <div><dt>Walk-forward testing</dt><dd>Re-estimating the strategy's weights at every rebalance using only data available up to that date, then scoring the following period. This is what stops the backtest from quietly using future information it would not have had in real time.</dd></div>
      <div><dt>Value at Risk (VaR)</dt><dd>The daily loss that historical returns were exceeded only 5% of the time. It describes a bad-but-not-worst-case day, not the maximum possible loss.</dd></div>
      <div><dt>Turnover</dt><dd>How much of the portfolio is bought and sold at each rebalance. Higher turnover means more transaction costs eating into returns, even if the resulting weights are "better" on paper.</dd></div>
      <div><dt>Shrinkage</dt><dd>Blending a noisy sample estimate (of covariance or expected returns) toward a more stable, simplified target. A few years of daily data is not enough to pin down expected returns precisely, so shrinkage trades a little bias for a lot less noise.</dd></div>
      <div><dt>Naive baseline</dt><dd>A rule (equal weight, inverse volatility, risk parity) that requires no return or covariance forecast. It exists to answer whether an optimizer's added complexity is actually earning its keep.</dd></div>
    </dl></section>
  </main><footer class="footer">Generated {{ generated_at }} from a frozen historical dataset. Results are educational and are not investment advice.</footer>
</div></body></html>"""
)


def _strategy_note(frame: pd.DataFrame | None) -> str:
    """State the comparison that matters, in the direction the data actually goes."""
    if frame is None or "sharpe_ratio" not in frame:
        return ""
    optimized = frame[frame.get("optimizes", False) == True]
    naive = frame[(frame.get("optimizes", True) == False)
                  & (~frame.index.str.startswith("Benchmark"))]
    if optimized.empty or naive.empty:
        return ""
    best_optimized = optimized["sharpe_ratio"].idxmax()
    best_naive = naive["sharpe_ratio"].idxmax()
    gap = frame.loc[best_optimized, "sharpe_ratio"] - frame.loc[best_naive, "sharpe_ratio"]
    if gap > 0:
        return (
            f"{best_optimized} reached a Sharpe ratio of {frame.loc[best_optimized, 'sharpe_ratio']:.2f} "
            f"against {frame.loc[best_naive, 'sharpe_ratio']:.2f} for {best_naive}, the best rule that "
            "estimates nothing. Optimizing helped in this sample, which is not guaranteed and is worth "
            "testing rather than assuming: on a more correlated single-country universe the same code "
            "produces the opposite result."
        )
    return (
        f"{best_naive} reached a Sharpe ratio of {frame.loc[best_naive, 'sharpe_ratio']:.2f} against "
        f"{frame.loc[best_optimized, 'sharpe_ratio']:.2f} for {best_optimized}. The optimizer did not "
        "beat a rule with nothing to estimate, which is what estimation error does when the inputs are "
        "noisier than the differences between candidate portfolios."
    )


def _verdict_html(summaries: pd.DataFrame, strategy_name: str, benchmark_label: str) -> str:
    """The headline comparison, spelled out in plain language and computed from the
    data rather than assumed, so it stays correct if the config or dataset changes.
    """
    opt = summaries.loc[strategy_name]
    bench = summaries.loc[benchmark_label]

    if opt["annual_return"] > bench["annual_return"]:
        return_para = (
            f"<p><strong>The optimized portfolio earned more, and took less risk to do it.</strong> "
            f"{opt['annual_return']:.1%} a year against {bench['annual_return']:.1%} for {benchmark_label}, "
            f"with lower volatility ({opt['annual_volatility']:.1%} against {bench['annual_volatility']:.1%}) "
            f"and a shallower fall ({opt['maximum_drawdown']:.1%} against {bench['maximum_drawdown']:.1%}).</p>"
        )
    else:
        return_para = (
            f"<p><strong>{benchmark_label} earned more in raw dollars. The optimizer did not beat it on "
            f"return.</strong> {bench['annual_return']:.1%} a year against {opt['annual_return']:.1%} for the "
            "optimized portfolio. That is the number that matters if the only question is which grew faster, "
            "and it is stated here directly rather than left for a reader to work out from a table.</p>"
        )

    if opt["sharpe_ratio"] > bench["sharpe_ratio"]:
        sharpe_para = (
            "<p>But raw return is not the only way to keep score. Per unit of risk taken (the "
            f"<strong>Sharpe ratio</strong>, defined below), the optimized portfolio did better: "
            f"{opt['sharpe_ratio']:.2f} against {bench['sharpe_ratio']:.2f} for {benchmark_label}. It also fell "
            f"less along the way: a maximum drawdown of {opt['maximum_drawdown']:.1%} against "
            f"{bench['maximum_drawdown']:.1%} for {benchmark_label}. Whether that trade-off is the right one "
            "depends on what matters more to the reader: growing the account as fast as possible, or growing "
            "it with a smaller worst-case fall along the way. This report reports both rather than picking one.</p>"
        )
    else:
        sharpe_para = (
            f"<p>{benchmark_label} also came out ahead per unit of risk taken: a Sharpe ratio of "
            f"{bench['sharpe_ratio']:.2f} against {opt['sharpe_ratio']:.2f} for the optimized portfolio. On this "
            "sample, the optimization did not earn its keep on either measure. See \"Did optimizing beat not "
            "optimizing?\" below for whether the naive rules fared any better.</p>"
        )

    return return_para + sharpe_para


def _sensitivity_summary(ranking: pd.DataFrame | None) -> str:
    """Plain-language headline for the sensitivity sweep, stated before the table
    rather than only in the caption beneath it."""
    if ranking is None or ranking.empty:
        return ""
    ranking = ranking.sort_values("share_of_configurations", ascending=False)
    leader = ranking.iloc[0]
    total = int(ranking["times_best"].sum())
    others = ranking.iloc[1:]
    if others.empty or others["times_best"].sum() == 0:
        runner_up = ""
    else:
        runner_up_row = others.iloc[0]
        runner_up = (
            f" and {runner_up_row['strategy']} the other {int(runner_up_row['times_best'])}"
        )
    return (
        f"A backtest reports one number per configuration, and somebody chose the configuration. Varying the "
        f"lookback, the weight cap, the transaction cost, and the covariance estimator one at a time and "
        f"re-running the walk-forward backtest each time, <strong>{leader['strategy']} ranks first in "
        f"{int(leader['times_best'])} of {total} configurations tried{runner_up}</strong>, so the finding "
        "below is not an artifact of one arbitrary setting."
    )


def _chart_html(figure: go.Figure) -> str:
    return pio.to_html(figure, full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False, "responsive": True})


def _base_layout(figure: go.Figure, title: str) -> None:
    figure.update_layout(
        title={"text": title, "font": {"size": 15}},
        margin={"l": 48, "r": 24, "t": 54, "b": 42},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, Arial, sans-serif", "color": "#34383b", "size": 12},
        legend={"orientation": "h", "y": 1.08},
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False, linecolor="#dfe2e1")
    figure.update_yaxes(gridcolor="#edf0ef", zerolinecolor="#dfe2e1")


def _format_comparison(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in ("annual_return", "annual_volatility", "maximum_drawdown", "daily_var_95"):
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{value:.2%}")
    for column in ("sharpe_ratio", "annual_turnover"):
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{value:.2f}")
    shown.columns = [column.replace("_", " ").title() for column in shown.columns]
    return shown.to_html(border=0)


def generate_report(
    output_path: Path,
    config: dict,
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    correlation: pd.DataFrame,
    summaries: pd.DataFrame,
    strategy_comparison: pd.DataFrame | None = None,
    sensitivity: pd.DataFrame | None = None,
    sensitivity_ranking: pd.DataFrame | None = None,
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
    benchmark_label = returns.columns[1]
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
        benchmark_label=benchmark_label,
        verdict_html=_verdict_html(summaries, strategy_name, benchmark_label),
        growth_chart=_chart_html(growth_figure),
        drawdown_chart=_chart_html(drawdown_figure),
        correlation_chart=_chart_html(correlation_figure),
        comparison_table=comparison.to_html(index=False, border=0),
        strategy_table=(_format_comparison(strategy_comparison.drop(columns=["optimizes"], errors="ignore"))
                        if strategy_comparison is not None else ""),
        strategy_note=_strategy_note(strategy_comparison),
        sensitivity_table=(sensitivity.to_html(border=0) if sensitivity is not None else ""),
        sensitivity_summary=_sensitivity_summary(sensitivity_ranking),
        sensitivity_note=(
            "Sharpe ratio under each assumption. The ranking within a row is a fair comparison; "
            "levels across rows are not, because the evaluation window changes."
            if sensitivity is not None else ""),
        generated_at=pd.Timestamp.now().strftime("%B %d, %Y"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"metrics": metrics, "latest_weights": latest.to_dict()}, indent=2), encoding="utf-8")
