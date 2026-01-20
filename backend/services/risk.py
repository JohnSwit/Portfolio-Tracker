import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from scipy import stats
import logging

from models.portfolio import Portfolio, RiskMetrics, StressTestScenario, StressTestResult
from services.market_data import market_data_service

logger = logging.getLogger(__name__)


class RiskService:
    """Service for calculating risk analytics"""

    def calculate_risk_metrics(
        self,
        portfolio: Portfolio,
        start_date: datetime,
        end_date: datetime,
        benchmark: str = "^GSPC",
        confidence_level: float = 0.95
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics

        Args:
            portfolio: Portfolio data
            start_date: Start date for analysis
            end_date: End date
            benchmark: Benchmark ticker
            confidence_level: Confidence level for VaR (default 95%)

        Returns:
            RiskMetrics object
        """
        try:
            # Get portfolio and benchmark returns
            symbols = [h.symbol for h in portfolio.holdings]
            price_data = market_data_service.get_price_data(symbols, start_date, end_date)

            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(
                portfolio, price_data, symbols
            )

            # Get benchmark returns
            benchmark_data = market_data_service.get_benchmark_data(benchmark, start_date, end_date)
            benchmark_returns = benchmark_data.pct_change().dropna()

            # Volatility measures
            daily_vol = portfolio_returns.std()
            annual_vol = daily_vol * np.sqrt(252)

            # Downside volatility (semi-deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252)

            # Risk-free rate
            risk_free_rate = market_data_service.get_risk_free_rate()
            daily_rf = risk_free_rate / 252

            # Sharpe ratio
            excess_returns = portfolio_returns - daily_rf
            sharpe_ratio = (excess_returns.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0.0

            # Sortino ratio
            sortino_ratio = (excess_returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0.0

            # Beta and Alpha
            aligned = pd.DataFrame({
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            }).dropna()

            if len(aligned) > 1:
                covariance = aligned.cov().iloc[0, 1]
                benchmark_variance = aligned['benchmark'].var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

                # Alpha (Jensen's alpha)
                portfolio_avg_return = aligned['portfolio'].mean() * 252
                benchmark_avg_return = aligned['benchmark'].mean() * 252
                alpha = portfolio_avg_return - (risk_free_rate + beta * (benchmark_avg_return - risk_free_rate))

                correlation = aligned['portfolio'].corr(aligned['benchmark'])
            else:
                beta = 1.0
                alpha = 0.0
                correlation = 0.0

            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max

            max_drawdown = drawdown.min()
            current_drawdown = drawdown.iloc[-1]

            # Max drawdown duration
            if len(drawdown) > 0:
                is_drawdown = drawdown < 0
                drawdown_periods = (is_drawdown != is_drawdown.shift()).cumsum()
                max_dd_duration = is_drawdown.groupby(drawdown_periods).sum().max() if is_drawdown.any() else 0
            else:
                max_dd_duration = 0

            # Calmar ratio
            annualized_return = portfolio_returns.mean() * 252
            calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0.0

            # Value at Risk (VaR)
            var_95 = self._calculate_var(portfolio_returns, 0.95)
            var_99 = self._calculate_var(portfolio_returns, 0.99)

            # Conditional VaR (Expected Shortfall)
            cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)

            # Factor exposures
            factor_exposures = self._calculate_factor_exposures(
                portfolio, portfolio_returns, start_date, end_date
            )

            period_label = f"{start_date.date()} to {end_date.date()}"

            return RiskMetrics(
                period=period_label,
                daily_volatility=daily_vol,
                annual_volatility=annual_vol,
                downside_volatility=downside_vol,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                beta=beta,
                alpha=alpha,
                correlation=correlation,
                max_drawdown=max_drawdown,
                max_drawdown_duration=int(max_dd_duration),
                current_drawdown=current_drawdown,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                factor_exposures=factor_exposures
            )

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            raise

    def _calculate_portfolio_returns(
        self,
        portfolio: Portfolio,
        price_data: pd.DataFrame,
        symbols: List[str]
    ) -> pd.Series:
        """Calculate portfolio returns from price data"""
        # Calculate weights
        total_value = sum(h.market_value or 0 for h in portfolio.holdings)
        weights = {h.symbol: (h.market_value or 0) / total_value
                  for h in portfolio.holdings if total_value > 0}

        # Calculate individual stock returns
        returns_df = pd.DataFrame()
        for symbol in symbols:
            if len(symbols) == 1:
                # For single symbol, yfinance returns different structure
                if 'Close' in price_data.columns:
                    prices = price_data['Close']
                else:
                    prices = price_data
                # Ensure 1D series
                if hasattr(prices, 'squeeze'):
                    prices = prices.squeeze()
            else:
                prices = price_data[symbol]['Close']

            returns_df[symbol] = prices.pct_change()

        # Weight the returns
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        for symbol in symbols:
            portfolio_returns += returns_df[symbol] * weights.get(symbol, 0.0)

        return portfolio_returns.dropna()

    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Value at Risk using historical simulation

        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as a positive number (loss)
        """
        if len(returns) == 0:
            return 0.0

        # Historical VaR
        var = -np.percentile(returns, (1 - confidence_level) * 100)
        return float(var)

    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)

        Average of returns worse than VaR

        Args:
            returns: Return series
            confidence_level: Confidence level

        Returns:
            CVaR as a positive number
        """
        if len(returns) == 0:
            return 0.0

        var = -self._calculate_var(returns, confidence_level)
        # Average of returns below VaR threshold
        cvar = -returns[returns <= var].mean()
        return float(cvar)

    def _calculate_factor_exposures(
        self,
        portfolio: Portfolio,
        portfolio_returns: pd.Series,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Calculate factor exposures using regression"""
        try:
            # Get factor returns
            factor_data = market_data_service.get_factor_data(start_date, end_date)

            # Align data
            aligned = pd.concat([portfolio_returns.rename('portfolio'), factor_data], axis=1).dropna()

            if len(aligned) < 10:  # Need sufficient data
                return {}

            # Run regression
            from sklearn.linear_model import LinearRegression

            X = aligned[['Market', 'Size', 'Value', 'Momentum', 'Quality']]
            y = aligned['portfolio']

            model = LinearRegression()
            model.fit(X, y)

            # Factor loadings
            exposures = {
                'Market': float(model.coef_[0]),
                'Size': float(model.coef_[1]),
                'Value': float(model.coef_[2]),
                'Momentum': float(model.coef_[3]),
                'Quality': float(model.coef_[4])
            }

            return exposures

        except Exception as e:
            logger.warning(f"Could not calculate factor exposures: {e}")
            return {}

    def stress_test(
        self,
        portfolio: Portfolio,
        scenario: StressTestScenario
    ) -> StressTestResult:
        """
        Run stress test scenario on portfolio

        Args:
            portfolio: Portfolio data
            scenario: Stress test scenario

        Returns:
            StressTestResult
        """
        try:
            current_value = portfolio.total_value or 0.0

            # Apply market shock if specified
            if scenario.market_shock is not None:
                estimated_loss_pct = scenario.market_shock
            else:
                # Calculate weighted impact based on factor shocks
                # This is simplified - in production would use factor exposures
                estimated_loss_pct = sum(scenario.factor_shocks.values()) / len(scenario.factor_shocks)

            estimated_loss = current_value * abs(estimated_loss_pct)
            portfolio_value_after = current_value - estimated_loss

            # Check if this exceeds VaR
            # Simplified check - would compare to actual VaR calculation
            var_breach = abs(estimated_loss_pct) > 0.05  # Exceeds typical 95% VaR

            return StressTestResult(
                scenario=scenario.name,
                estimated_loss=estimated_loss,
                estimated_loss_pct=estimated_loss_pct,
                portfolio_value_after=portfolio_value_after,
                var_breach=var_breach
            )

        except Exception as e:
            logger.error(f"Error running stress test: {e}")
            raise

    def run_scenario_analysis(
        self,
        portfolio: Portfolio,
        scenarios: List[StressTestScenario]
    ) -> List[StressTestResult]:
        """Run multiple stress test scenarios"""
        results = []
        for scenario in scenarios:
            result = self.stress_test(portfolio, scenario)
            results.append(result)
        return results


# Predefined stress scenarios
STRESS_SCENARIOS = [
    StressTestScenario(
        name="2008 Financial Crisis",
        description="Market crash scenario similar to 2008",
        factor_shocks={'Market': -0.37, 'Value': -0.25, 'Size': -0.30},
        market_shock=-0.37
    ),
    StressTestScenario(
        name="COVID-19 Crash",
        description="Market drop similar to March 2020",
        factor_shocks={'Market': -0.34, 'Value': -0.28, 'Growth': -0.15},
        market_shock=-0.34
    ),
    StressTestScenario(
        name="Tech Bubble Burst",
        description="Tech sector crash",
        factor_shocks={'Market': -0.20, 'Growth': -0.40, 'Momentum': -0.35},
        market_shock=-0.25
    ),
    StressTestScenario(
        name="Rising Interest Rates",
        description="Rapid rate increase scenario",
        factor_shocks={'Market': -0.15, 'Growth': -0.20, 'Value': 0.05},
        market_shock=-0.12
    ),
    StressTestScenario(
        name="Moderate Correction",
        description="Standard 10% market correction",
        factor_shocks={'Market': -0.10},
        market_shock=-0.10
    )
]


# Global instance
risk_service = RiskService()
