WITH monthly_prices AS (
    SELECT
        ticker,
        substr(date, 1, 7) AS month,
        adjusted_close,
        ROW_NUMBER() OVER (
            PARTITION BY ticker, substr(date, 1, 7)
            ORDER BY date DESC
        ) AS observation_rank
    FROM daily_prices
),
month_end_prices AS (
    SELECT ticker, month, adjusted_close
    FROM monthly_prices
    WHERE observation_rank = 1
),
monthly_returns AS (
    SELECT
        ticker,
        month,
        adjusted_close / LAG(adjusted_close) OVER (
            PARTITION BY ticker ORDER BY month
        ) - 1.0 AS monthly_return
    FROM month_end_prices
),
rolling_risk AS (
    SELECT
        ticker,
        month,
        monthly_return,
        AVG(monthly_return) OVER (
            PARTITION BY ticker ORDER BY month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) AS rolling_12m_average_return
    FROM monthly_returns
)
SELECT
    r.ticker,
    a.asset_class,
    r.month,
    r.monthly_return,
    r.rolling_12m_average_return,
    RANK() OVER (
        PARTITION BY r.month ORDER BY ABS(r.monthly_return) DESC
    ) AS absolute_move_rank
FROM rolling_risk r
JOIN assets a ON a.ticker = r.ticker
WHERE r.monthly_return IS NOT NULL
ORDER BY r.month, absolute_move_rank;
