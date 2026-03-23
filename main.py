"""
Main Bot — Continuous Crypto Trading Bot for Roostoo Hackathon
===============================================================
Autonomous 24/7 trading bot that:
1. Fetches market data (Binance historical + Roostoo live prices)
2. Computes alpha signals (momentum, mean-reversion, regime, etc.)
3. Optimizes portfolio (Black-Litterman + HRP)
4. Executes rebalancing trades via Roostoo API
5. Monitors risk (drawdown, stops, correlation spikes)
6. Logs everything for audit trail

Usage:
    python main.py                                          # Run bot (continuous loop)
    python main.py --once                                   # Single rebalance cycle (live)
    python main.py --status                                 # Check portfolio status
    python main.py --test                                   # Test API connectivity
    python main.py --backtest --start 2025-01-01 --end 2025-03-01  # Backtest
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from roostoo_client import RoostooClient
from data_engine import DataEngine
from signals import AlphaEngine
from optimizer import PortfolioOptimizer
from executor import TradeExecutor, TradeOrder
from risk_manager import RiskManager

logger = logging.getLogger(__name__)


class RoostooSpecClient(RoostooClient):
    """Roostoo client adapter that normalizes README2 API response shapes.

    README2 defines ticker payloads under "Data" and trade payloads under
    "OrderDetail". The bot/executor pipeline expects flatter dicts, so this
    adapter bridges those formats without changing trading logic.
    """

    def ticker(self, pair: str = None) -> Dict[str, Dict[str, Any]]:
        """Return ticker map keyed by pair, regardless of raw response shape."""
        raw = super().ticker(pair=pair)
        if not isinstance(raw, dict):
            return {}

        if "Data" in raw and isinstance(raw["Data"], dict):
            return raw["Data"]

        # Fallback for non-wrapped payloads.
        return raw

    def get_all_prices(self) -> Dict[str, float]:
        """Get latest prices from ticker map with README2-compatible parsing."""
        data = self.ticker()
        prices: Dict[str, float] = {}
        for pair_name, ticker_data in data.items():
            if not isinstance(ticker_data, dict):
                continue
            last_price = ticker_data.get("LastPrice")
            if last_price is None:
                continue
            try:
                prices[pair_name] = float(last_price)
            except (TypeError, ValueError):
                continue
        return prices

    def place_order(self, pair: str, side: str, quantity: float,
                    order_type: str = "MARKET", price: float = None) -> Dict:
        """Place order and return a flattened order dict for executor compatibility."""
        raw = super().place_order(
            pair=pair,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
        )

        if not isinstance(raw, dict):
            return {}

        detail = raw.get("OrderDetail")
        if not isinstance(detail, dict):
            return raw

        # Preserve legacy key access expected by executor while keeping raw fields.
        normalized = dict(detail)
        normalized["Success"] = raw.get("Success")
        normalized["ErrMsg"] = raw.get("ErrMsg", "")
        if "FilledAverPrice" not in normalized and "Price" in normalized:
            normalized["FilledAverPrice"] = normalized.get("Price", 0)
        return normalized

    def query_orders(self, order_id: int = None, pair: str = None,
                     pending_only: bool = False, limit: int = 100) -> List[Dict]:
        """Return list of matched orders from README2 query payload shape."""
        data = {"limit": limit}
        if order_id:
            data["order_id"] = order_id
        elif pair:
            data["pair"] = pair
            if pending_only:
                data["pending_only"] = "TRUE"

        result = self._post("/v3/query_order", data)
        if isinstance(result, dict):
            matched = result.get("OrderMatched", [])
            if isinstance(matched, list):
                return matched
        if isinstance(result, list):
            return result
        return []

# ── Graceful shutdown ──
_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received (signal %d), finishing current cycle...", signum)
    _shutdown = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── State persistence ──

def load_state() -> Dict:
    """Load bot state from disk."""
    if os.path.exists(cfg.STATE_FILE):
        try:
            with open(cfg.STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load state: %s", e)
    return {
        "last_rebalance": None,
        "last_processed_bar": None,
        "rebalance_count": 0,
        "nav_history": [],
        "entry_prices": {},
        "stop_cooldown_until": {},
        "total_trades": 0,
        "total_commission": 0.0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def save_state(state: Dict):
    """Persist bot state to disk."""
    os.makedirs(os.path.dirname(cfg.STATE_FILE), exist_ok=True)
    with open(cfg.STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _compute_average_pairwise_correlation(returns: pd.DataFrame, lookback: int = 24) -> float:
    """Compute recent average pairwise correlation across assets."""
    if returns is None or returns.empty:
        return 0.0
    if len(returns) < 2:
        return 0.0

    recent = returns.iloc[-min(lookback, len(returns)):]
    corr = recent.corr()
    n = len(corr)
    if n < 2:
        return 0.0

    mask = ~np.eye(n, dtype=bool)
    values = corr.values[mask]
    if values.size == 0:
        return 0.0
    return float(np.nanmean(values))


def _compute_composite_score(risk_summary: Dict[str, float]) -> float:
    """Compute hackathon composite risk-adjusted score from summary metrics."""
    sortino = float(risk_summary.get("sortino", 0.0) or 0.0)
    sharpe = float(risk_summary.get("sharpe", 0.0) or 0.0)
    calmar = float(risk_summary.get("calmar", 0.0) or 0.0)
    return 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar


def _signal_consensus_count(
    signal_scores: Dict[str, pd.Series],
    coin: str,
    direction: int,
    min_abs_signal: float = 0.5,
) -> int:
    """Count how many component signals agree with the target direction."""
    consensus = 0
    for signal in signal_scores.values():
        if coin not in signal.index:
            continue
        value = float(signal[coin])
        if abs(value) < min_abs_signal:
            continue
        if value * direction > 0:
            consensus += 1
    return consensus


def _reward_risk_ratio(
    alpha_value: float,
    consensus_count: int,
    composite_score: float,
    current_drawdown: float,
    avg_corr: float,
    alpha_threshold: float,
) -> float:
    """Reward-vs-risk score used to gate cooldown overrides."""
    reward_quality = (
        1.0
        + max(0.0, composite_score)
        + 0.20 * consensus_count
        + 0.30 * max(0.0, abs(alpha_value) - alpha_threshold)
    )
    risk_pressure = 1.0 + 2.5 * max(0.0, current_drawdown) + 2.0 * max(0.0, avg_corr - 0.5)
    return reward_quality / max(risk_pressure, 1e-9)


# ── Logging setup ──

def setup_logging():
    """Configure logging to both file and console."""
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    log_file = os.path.join(cfg.LOG_DIR, f"bot_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ── Trade logging ──

def log_trades_to_csv(orders: list, cycle_time: str):
    """Append executed trades to CSV for audit trail."""
    os.makedirs(cfg.CSV_DIR, exist_ok=True)
    csv_path = os.path.join(cfg.CSV_DIR, "trade_log.csv")

    rows = []
    for o in orders:
        rows.append({
            "timestamp": cycle_time,
            "pair": o.pair,
            "side": o.side,
            "type": o.order_type,
            "quantity": o.quantity,
            "fill_quantity": o.fill_quantity,
            "fill_price": o.fill_price,
            "commission": o.commission,
            "status": o.status,
            "order_id": o.order_id,
            "reason": o.reason,
        })

    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)


def log_portfolio_snapshot(nav: float, holdings: Dict, weights: Dict,
                           regime: str, risk_summary: Dict, cycle_time: str):
    """Append portfolio snapshot to CSV."""
    os.makedirs(cfg.CSV_DIR, exist_ok=True)
    csv_path = os.path.join(cfg.CSV_DIR, "portfolio_log.csv")

    row = {
        "timestamp": cycle_time,
        "nav_usd": nav,
        "regime": regime,
        "n_positions": len(holdings),
        "invested_pct": sum(weights.values()) if weights else 0,
        "max_position_pct": max(weights.values()) if weights else 0,
        "drawdown": risk_summary.get("current_drawdown", 0),
        "max_drawdown": risk_summary.get("max_drawdown", 0),
        "sortino": risk_summary.get("sortino", 0),
        "sharpe": risk_summary.get("sharpe", 0),
        "calmar": risk_summary.get("calmar", 0),
    }

    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False)


# ── Core bot logic ──

class TradingBot:
    """Main trading bot orchestrating all components."""

    def __init__(self):
        self.client = RoostooSpecClient()
        self.data_engine = DataEngine()
        live_rebalance_hours = 1 if cfg.REBALANCE.continuous_on_hour_bar else cfg.REBALANCE.frequency_hours
        self.alpha_engine = AlphaEngine(rebalance_hours=live_rebalance_hours)
        self.optimizer = PortfolioOptimizer()
        self.executor = TradeExecutor(self.client)
        self.risk_mgr = RiskManager()
        self.state = load_state()

        # Stop-loss cooldown: {coin: cooldown_expiry_utc}
        self._stop_cooldown: Dict[str, datetime] = {}
        for coin, expiry in self.state.get("stop_cooldown_until", {}).items():
            try:
                dt = datetime.fromisoformat(expiry)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                self._stop_cooldown[coin] = dt
            except Exception:
                continue

        # Restore risk manager state
        for nav in self.state.get("nav_history", []):
            self.risk_mgr.update_nav(nav)

    def _current_decision_bar(self) -> datetime:
        """Return latest fully closed hourly bar timestamp in UTC."""
        now = datetime.now(timezone.utc)
        return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

    def _refresh_cooldowns(self, now_ts: datetime):
        """Drop expired cooldown entries."""
        for coin in list(self._stop_cooldown):
            if self._stop_cooldown[coin] <= now_ts:
                del self._stop_cooldown[coin]

    def _set_cooldown(self, coin: str, now_ts: datetime):
        """Set cooldown expiry in wall-clock time."""
        self._stop_cooldown[coin] = now_ts + timedelta(hours=cfg.OUTLIER.cooldown_hours)

    def _is_on_cooldown(self, coin: str, now_ts: datetime) -> bool:
        expiry = self._stop_cooldown.get(coin)
        return bool(expiry and expiry > now_ts)

    def _serialize_cooldowns(self) -> Dict[str, str]:
        return {coin: dt.isoformat() for coin, dt in self._stop_cooldown.items()}

    def _apply_outlier_overrides(
        self,
        alpha_scores: pd.Series,
        signal_scores: Dict[str, pd.Series],
        returns: pd.DataFrame,
        now_ts: datetime,
    ):
        """Allow cooldown breaks when outlier signals have favorable reward-vs-risk."""
        if not cfg.OUTLIER.enabled:
            return

        risk_summary = self.risk_mgr.get_risk_summary()
        composite = _compute_composite_score(risk_summary)
        avg_corr = _compute_average_pairwise_correlation(returns)
        drawdown = float(self.risk_mgr.current_drawdown)

        for coin in list(self._stop_cooldown):
            if not self._is_on_cooldown(coin, now_ts):
                continue
            if coin not in alpha_scores.index:
                continue

            alpha_value = float(alpha_scores[coin])
            if abs(alpha_value) < cfg.OUTLIER.alpha_z_threshold:
                continue

            direction = 1 if alpha_value > 0 else -1
            consensus = _signal_consensus_count(signal_scores, coin, direction)
            if consensus < cfg.OUTLIER.min_signal_consensus:
                continue

            rr_ratio = _reward_risk_ratio(
                alpha_value=alpha_value,
                consensus_count=consensus,
                composite_score=composite,
                current_drawdown=drawdown,
                avg_corr=avg_corr,
                alpha_threshold=cfg.OUTLIER.alpha_z_threshold,
            )
            if rr_ratio < cfg.OUTLIER.min_reward_risk_ratio:
                continue

            if direction > 0:
                if drawdown > cfg.OUTLIER.max_drawdown_for_bull_override:
                    continue
                if avg_corr > cfg.OUTLIER.max_corr_for_bull_override:
                    continue
            elif (not cfg.OUTLIER.allow_bear_override_in_corr_spike
                  and avg_corr > cfg.RISK.correlation_spike_threshold):
                continue

            del self._stop_cooldown[coin]
            if direction < 0:
                alpha_scores[coin] = min(alpha_scores[coin], -10.0)
            logger.warning(
                "Cooldown override for %s | dir=%s alpha=%.2f consensus=%d rr=%.2f dd=%.2f%% corr=%.2f",
                coin,
                "BULL" if direction > 0 else "BEAR",
                alpha_value,
                consensus,
                rr_ratio,
                drawdown * 100,
                avg_corr,
            )

    def should_rebalance(self, decision_bar: Optional[datetime] = None) -> bool:
        """Check if a new decision event is available."""
        if cfg.REBALANCE.continuous_on_hour_bar:
            bar = decision_bar or self._current_decision_bar()
            last_bar_raw = self.state.get("last_processed_bar")
            if last_bar_raw is None:
                return True
            try:
                last_bar = datetime.fromisoformat(last_bar_raw)
                if last_bar.tzinfo is None:
                    last_bar = last_bar.replace(tzinfo=timezone.utc)
            except Exception:
                return True
            return bar > last_bar

        last = self.state.get("last_rebalance")
        if last is None:
            return True

        last_time = datetime.fromisoformat(last)
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
        required = cfg.REBALANCE.frequency_hours * 3600
        return elapsed >= required

    def _validate_trading_pairs(self):
        """Filter configured trading pairs to those actually available on exchange."""
        try:
            exchange_pairs = self.client.get_trading_pairs()
            available = set(exchange_pairs.keys())
            configured = set(cfg.TRADING_PAIRS)

            valid = sorted(configured & available)
            missing = configured - available

            if missing:
                logger.warning("Pairs NOT available on Roostoo (removed from universe): %s",
                               sorted(missing))

            # Update config in-place
            cfg.TRADING_PAIRS[:] = valid
            cfg.COINS[:] = [p.split("/")[0] for p in valid]
            # Update Binance mapping
            cfg.ROOSTOO_TO_BINANCE.clear()
            cfg.ROOSTOO_TO_BINANCE.update({
                f"{coin}/USD": f"{coin}USDT" for coin in cfg.COINS
            })

            logger.info("Active trading universe (%d pairs): %s", len(valid), valid)
        except Exception as e:
            logger.warning("Could not validate trading pairs: %s", e)

    def test_connectivity(self) -> bool:
        """Test API connectivity and print exchange info."""
        try:
            server_time = self.client.server_time()
            logger.info("Server time: %s", datetime.fromtimestamp(server_time / 1000, tz=timezone.utc))

            if not self.client.is_exchange_running():
                logger.warning("Exchange is NOT running")
                return False

            info = self.client.exchange_info()
            pairs = info.get("TradePairs", {})
            logger.info("Available pairs (%d): %s", len(pairs), list(pairs.keys()))

            # Validate which of our configured pairs actually exist
            self._validate_trading_pairs()

            wallet = self.client.balance()
            logger.info("Wallet: %s", json.dumps(wallet, indent=2))

            if not wallet:
                try:
                    raw_balance = self.client._get("/v3/balance", signed=True)
                    logger.warning(
                        "Balance endpoint returned empty parsed wallet. Raw keys=%s Success=%s ErrMsg=%s",
                        list(raw_balance.keys()) if isinstance(raw_balance, dict) else type(raw_balance).__name__,
                        raw_balance.get("Success") if isinstance(raw_balance, dict) else None,
                        raw_balance.get("ErrMsg") if isinstance(raw_balance, dict) else None,
                    )
                except Exception as diag_err:
                    logger.warning("Failed to fetch raw balance diagnostics: %s", diag_err)

            nav = self.client.get_portfolio_value()
            logger.info("Portfolio value: $%.2f", nav)
            if nav <= 0:
                logger.warning(
                    "Connected successfully, but wallet appears unfunded (NAV <= 0). "
                    "Bot will skip live trading until account has USD/coin balance."
                )

            return True
        except Exception as e:
            logger.error("Connectivity test failed: %s", e)
            return False

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status without rebalancing."""
        try:
            wallet = self.client.balance()
            prices = self.client.get_all_prices()
            nav = self.client.get_portfolio_value()
            holdings = self.executor.get_current_holdings()
            usd = self.executor.get_usd_balance()

            # Compute weights
            weights = {}
            for coin, qty in holdings.items():
                pair = f"{coin}/USD"
                if pair in prices and nav > 0:
                    weights[coin] = (qty * prices[pair]) / nav

            risk_summary = self.risk_mgr.get_risk_summary()

            return {
                "nav": nav,
                "usd_free": usd,
                "holdings": holdings,
                "weights": weights,
                "prices": prices,
                "regime": self.alpha_engine.regime_name,
                "risk": risk_summary,
                "rebalance_count": self.state.get("rebalance_count", 0),
                "total_trades": self.state.get("total_trades", 0),
                "total_commission": self.state.get("total_commission", 0),
            }
        except Exception as e:
            logger.error("Failed to get status: %s", e)
            return {"error": str(e)}

    def run_rebalance_cycle(self, decision_bar: Optional[datetime] = None) -> Dict:
        """Execute one full rebalance cycle.

        Returns:
            Dict with cycle results (nav, orders, regime, etc.)
        """
        event_ts = decision_bar or datetime.now(timezone.utc)
        cycle_time = event_ts.isoformat()
        logger.info("=" * 60)
        logger.info("Starting rebalance cycle at %s", cycle_time)

        # 1. Check exchange status
        if not self.client.is_exchange_running():
            logger.warning("Exchange not running, skipping cycle")
            return {"status": "SKIPPED", "reason": "exchange_not_running"}

        # 2. Cancel any pending orders
        self.executor.cancel_all_pending()

        # 3. Fetch current state
        holdings = self.executor.get_current_holdings()
        prices = self.client.get_all_prices()
        nav = self.client.get_portfolio_value()
        usd_balance = self.executor.get_usd_balance()

        logger.info("NAV: $%.2f | USD: $%.2f | Positions: %d",
                     nav, usd_balance, len(holdings))

        if nav <= 0 or (usd_balance <= 0 and len(holdings) == 0):
            try:
                raw_balance = self.client._get("/v3/balance", signed=True)
                logger.warning(
                    "Balance diagnostics: keys=%s Success=%s ErrMsg=%s",
                    list(raw_balance.keys()) if isinstance(raw_balance, dict) else type(raw_balance).__name__,
                    raw_balance.get("Success") if isinstance(raw_balance, dict) else None,
                    raw_balance.get("ErrMsg") if isinstance(raw_balance, dict) else None,
                )
            except Exception as diag_err:
                logger.warning("Balance diagnostics failed: %s", diag_err)

            logger.warning(
                "No funded wallet detected (NAV=$%.2f, USD=$%.2f, positions=%d). "
                "Skipping cycle; no orders can be sized without capital.",
                nav,
                usd_balance,
                len(holdings),
            )
            return {
                "status": "SKIPPED",
                "reason": "no_funded_wallet",
                "nav": nav,
                "usd_balance": usd_balance,
                "positions": len(holdings),
            }

        # Update risk tracking
        self.risk_mgr.update_nav(nav)

        # 4. Fetch historical data
        close_prices = self.data_engine.get_close_prices(
            interval="1h",
            days=cfg.SIGNALS.regime_lookback_hours // 24 + 5,
        )
        returns = self.data_engine.get_returns(
            interval="1h",
            days=cfg.SIGNALS.regime_lookback_hours // 24 + 5,
        )

        if close_prices.empty or returns.empty:
            logger.warning("No historical data available, skipping cycle")
            return {"status": "SKIPPED", "reason": "no_data"}

        # Fetch volume data for VolumeMomentumSignal
        volumes = self.data_engine.get_volumes(
            interval="1h",
            days=cfg.SIGNALS.regime_lookback_hours // 24 + 5,
        )

        # 5. Compute alpha signals
        alpha_scores = self.alpha_engine.compute_alpha(close_prices, returns, volumes=volumes)
        regime = self.alpha_engine.current_regime
        regime_name = self.alpha_engine.regime_name

        logger.info("Regime: %s | Alpha scores: %s",
                     regime_name, alpha_scores.to_dict())

        # 6. Check risk conditions

        # Expire elapsed cooldowns using wall-clock time.
        self._refresh_cooldowns(event_ts)

        # Widen stop-loss dynamically using ATR-based stops
        # (pass returns so RiskManager can compute per-asset realized vol)
        stop_pct = cfg.RISK.position_stop_loss_pct  # fallback only

        # Only check positions not on cooldown
        check_coins = {c for c in holdings if not self._is_on_cooldown(c, event_ts)}
        stopped = self.risk_mgr.check_position_stops(
            {f"{c}/USD": prices.get(f"{c}/USD", 0) for c in check_coins},
            {f"{c}/USD": self.state.get("entry_prices", {}).get(c, 0) for c in check_coins},
            stop_pct_override=stop_pct,
            returns=returns,
        )
        if stopped:
            for pair in stopped:
                coin = pair.split("/")[0]
                if coin in alpha_scores.index:
                    alpha_scores[coin] = -10.0  # strong sell signal
                self._set_cooldown(coin, event_ts)
                self.risk_mgr.reset_position_hwm(coin)

        # Prevent re-entry for coins on cooldown
        for coin in list(self._stop_cooldown):
            if coin in alpha_scores.index:
                alpha_scores[coin] = min(alpha_scores[coin], -1.0)

        # Break cooldown for extreme outliers when reward-vs-risk is favorable.
        self._apply_outlier_overrides(
            alpha_scores=alpha_scores,
            signal_scores=self.alpha_engine.last_signal_scores,
            returns=returns,
            now_ts=event_ts,
        )

        # Check correlation spike (crash detection)
        if self.risk_mgr.check_correlation_spike(returns):
            logger.warning("Correlation spike detected — shifting to defensive")
            regime = 2  # Force bear regime

        # 7. Optimize portfolio
        target_weights, diagnostics = self.optimizer.optimize(
            alpha_scores=alpha_scores,
            returns=returns,
            regime=regime,
            current_drawdown=self.risk_mgr.current_drawdown,
        )

        logger.info("Target weights: %s", target_weights.to_dict())
        logger.info("Diagnostics: %s", diagnostics)

        # 8. Generate orders
        target_dict = target_weights.to_dict()
        orders = self.executor.generate_orders(
            current_holdings=holdings,
            target_weights=target_dict,
            current_prices=prices,
            portfolio_value=nav,
        )

        if not orders:
            logger.info("No trades needed — portfolio already aligned")
            self.state["last_rebalance"] = cycle_time
            self.state["last_processed_bar"] = event_ts.isoformat()
            self.state["stop_cooldown_until"] = self._serialize_cooldowns()
            save_state(self.state)
            return {"status": "NO_TRADES", "nav": nav, "regime": regime_name}

        # Check turnover limit
        turnover = self.executor.compute_turnover(orders, prices, nav)
        if turnover["turnover_pct"] > cfg.REBALANCE.max_turnover_pct:
            logger.warning("Turnover %.1f%% exceeds limit %.1f%%, scaling down",
                           turnover["turnover_pct"] * 100,
                           cfg.REBALANCE.max_turnover_pct * 100)
            # Scale down order quantities proportionally
            scale = cfg.REBALANCE.max_turnover_pct / turnover["turnover_pct"]
            for o in orders:
                o.quantity = self.client.round_quantity(o.pair, o.quantity * scale)
            orders = [o for o in orders if o.quantity > 0]

        # 9. Execute orders
        logger.info("Executing %d orders...", len(orders))
        executed = self.executor.execute_orders(orders)

        # 10. Update state
        filled = [o for o in executed if o.status == "FILLED"]
        total_commission = sum(o.commission or 0 for o in filled)

        # Update entry prices for new buys
        entry_prices = self.state.get("entry_prices", {})
        for o in filled:
            coin = o.pair.split("/")[0]
            if o.side == "BUY" and o.fill_price:
                if coin not in entry_prices:
                    entry_prices[coin] = o.fill_price
            elif o.side == "SELL":
                # Check if fully sold
                new_holdings = self.executor.get_current_holdings()
                if coin not in new_holdings:
                    entry_prices.pop(coin, None)
                    self.risk_mgr.reset_position_hwm(coin)

        self.state.update({
            "last_rebalance": cycle_time,
            "last_processed_bar": event_ts.isoformat(),
            "rebalance_count": self.state.get("rebalance_count", 0) + 1,
            "entry_prices": entry_prices,
            "total_trades": self.state.get("total_trades", 0) + len(filled),
            "total_commission": self.state.get("total_commission", 0) + total_commission,
            "nav_history": self.risk_mgr.nav_history[-500:],  # keep last 500
            "stop_cooldown_until": self._serialize_cooldowns(),
        })
        save_state(self.state)

        # 11. Log to CSV
        log_trades_to_csv(executed, cycle_time)

        risk_summary = self.risk_mgr.get_risk_summary()
        log_portfolio_snapshot(
            nav=nav,
            holdings=holdings,
            weights=target_dict,
            regime=regime_name,
            risk_summary=risk_summary,
            cycle_time=cycle_time,
        )

        logger.info("Cycle complete: %d/%d orders filled | Commission: $%.4f",
                     len(filled), len(executed), total_commission)
        logger.info("Risk: DD=%.1f%% | Sortino=%.2f | Sharpe=%.2f | Calmar=%.2f",
                     risk_summary.get("current_drawdown", 0) * 100,
                     risk_summary.get("sortino", 0),
                     risk_summary.get("sharpe", 0),
                     risk_summary.get("calmar", 0))

        return {
            "status": "OK",
            "nav": nav,
            "regime": regime_name,
            "orders_total": len(executed),
            "orders_filled": len(filled),
            "commission": total_commission,
            "risk": risk_summary,
        }

    def run_loop(self):
        """Main continuous trading loop."""
        logger.info("Starting trading bot — continuous mode")
        if cfg.REBALANCE.continuous_on_hour_bar:
            logger.info("Decision mode: event-driven on each new closed 1h bar")
        else:
            logger.info("Decision mode: every %d hours", cfg.REBALANCE.frequency_hours)

        # Initial connectivity test + pair validation
        if not self.test_connectivity():
            logger.error("Initial connectivity test failed, exiting")
            sys.exit(1)
        self._validate_trading_pairs()

        while not _shutdown:
            try:
                decision_bar = self._current_decision_bar()
                if self.should_rebalance(decision_bar):
                    result = self.run_rebalance_cycle(decision_bar=decision_bar)
                    logger.info("Cycle result: %s", result.get("status"))
                else:
                    # Between cycles: just track NAV
                    try:
                        nav = self.client.get_portfolio_value()
                        self.risk_mgr.update_nav(nav)
                        logger.debug("NAV check: $%.2f (DD: %.1f%%)",
                                     nav, self.risk_mgr.current_drawdown * 100)
                    except Exception as e:
                        logger.debug("NAV check failed: %s", e)

            except Exception as e:
                logger.error("Rebalance cycle failed: %s", e, exc_info=True)

            # Sleep between event checks.
            sleep_seconds = max(1, int(cfg.REBALANCE.poll_seconds))
            for _ in range(sleep_seconds):
                if _shutdown:
                    break
                time.sleep(1)

        logger.info("Bot shutdown complete")
        save_state(self.state)


# ── Backtesting Engine ──

class BacktestEngine:
    """Backtest the signal -> optimize -> execute pipeline on historical data.

    Uses Binance hourly OHLCV data. Simulates rebalancing at close prices
    every N hours (matching cfg.REBALANCE.frequency_hours).

    Usage:
        engine = BacktestEngine(start="2025-01-01", end="2025-03-01")
        results = engine.run()
    """

    # Commission rate matching Roostoo (maker 0.008% + taker 0.012%)
    COMMISSION_RATE = 0.0001  # 0.01% per side (Roostoo is very low fee)

    def __init__(self, start: str, end: str, initial_capital: float = None,
                 rebalance_hours: int = None, adaptive: bool = True):
        self.start = pd.Timestamp(start, tz="UTC")
        self.end = pd.Timestamp(end, tz="UTC")
        self.initial_capital = initial_capital or cfg.INITIAL_CAPITAL_USD
        self.rebalance_hours = rebalance_hours or cfg.REBALANCE.frequency_hours
        self.continuous_mode = bool(cfg.REBALANCE.continuous_on_hour_bar and rebalance_hours is None)
        engine_rebalance_hours = 1 if self.continuous_mode else self.rebalance_hours

        self.data_engine = DataEngine()
        self.alpha_engine = AlphaEngine(adaptive=adaptive,
                        rebalance_hours=engine_rebalance_hours)
        self.optimizer = PortfolioOptimizer()
        self.risk_mgr = RiskManager()

        # State
        self.cash = self.initial_capital
        self.holdings: Dict[str, float] = {}  # {coin: quantity}
        self.entry_prices: Dict[str, float] = {}

        # Stop-loss cooldown: {coin: cooldown_expiry_ts}
        self._stop_cooldown: Dict[str, pd.Timestamp] = {}

        # Results tracking
        self.nav_series: List[Tuple[pd.Timestamp, float]] = []
        self.trade_log: List[Dict] = []
        self.portfolio_log: List[Dict] = []
        self._volume_cache: Dict[str, pd.Series] = {}  # coin -> volume series

    def _refresh_cooldowns(self, now_ts: pd.Timestamp):
        for coin in list(self._stop_cooldown):
            if self._stop_cooldown[coin] <= now_ts:
                del self._stop_cooldown[coin]

    def _set_cooldown(self, coin: str, now_ts: pd.Timestamp):
        self._stop_cooldown[coin] = now_ts + pd.Timedelta(hours=cfg.OUTLIER.cooldown_hours)

    def _is_on_cooldown(self, coin: str, now_ts: pd.Timestamp) -> bool:
        expiry = self._stop_cooldown.get(coin)
        return bool(expiry is not None and expiry > now_ts)

    def _apply_outlier_overrides(
        self,
        alpha_scores: pd.Series,
        signal_scores: Dict[str, pd.Series],
        returns: pd.DataFrame,
        now_ts: pd.Timestamp,
    ):
        if not cfg.OUTLIER.enabled:
            return

        risk_summary = self.risk_mgr.get_risk_summary()
        composite = _compute_composite_score(risk_summary)
        avg_corr = _compute_average_pairwise_correlation(returns)
        drawdown = float(self.risk_mgr.current_drawdown)

        for coin in list(self._stop_cooldown):
            if not self._is_on_cooldown(coin, now_ts):
                continue
            if coin not in alpha_scores.index:
                continue

            alpha_value = float(alpha_scores[coin])
            if abs(alpha_value) < cfg.OUTLIER.alpha_z_threshold:
                continue

            direction = 1 if alpha_value > 0 else -1
            consensus = _signal_consensus_count(signal_scores, coin, direction)
            if consensus < cfg.OUTLIER.min_signal_consensus:
                continue

            rr_ratio = _reward_risk_ratio(
                alpha_value=alpha_value,
                consensus_count=consensus,
                composite_score=composite,
                current_drawdown=drawdown,
                avg_corr=avg_corr,
                alpha_threshold=cfg.OUTLIER.alpha_z_threshold,
            )
            if rr_ratio < cfg.OUTLIER.min_reward_risk_ratio:
                continue

            if direction > 0:
                if drawdown > cfg.OUTLIER.max_drawdown_for_bull_override:
                    continue
                if avg_corr > cfg.OUTLIER.max_corr_for_bull_override:
                    continue
            elif (not cfg.OUTLIER.allow_bear_override_in_corr_spike
                  and avg_corr > cfg.RISK.correlation_spike_threshold):
                continue

            del self._stop_cooldown[coin]
            if direction < 0:
                alpha_scores[coin] = min(alpha_scores[coin], -10.0)

    def _fetch_historical_range(self, coins: List[str], start: pd.Timestamp,
                                end: pd.Timestamp, interval: str = "1h") -> pd.DataFrame:
        """Fetch historical close prices for a specific date range from Binance.

        Unlike DataEngine.get_close_prices (which fetches N days from now),
        this fetches data for an exact start->end window.
        """
        from data_engine import BinanceDataClient
        binance = BinanceDataClient()

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        close_frames = {}
        for coin in coins:
            binance_symbol = cfg.ROOSTOO_TO_BINANCE.get(f"{coin}/USD", f"{coin}USDT")

            # Paginate through the range
            all_data = []
            current_start = start_ms
            while current_start < end_ms:
                df = binance.fetch_klines(
                    binance_symbol, interval,
                    start_time=current_start, end_time=end_ms, limit=1000,
                )
                if df.empty:
                    break
                all_data.append(df)
                last_ts = int(df.index[-1].timestamp() * 1000)
                if last_ts <= current_start:
                    break
                current_start = last_ts + 1
                time.sleep(0.05)

            if all_data:
                combined = pd.concat(all_data)
                combined = combined[~combined.index.duplicated(keep="last")]
                close_frames[coin] = combined["close"]
                if "volume" in combined.columns:
                    self._volume_cache[coin] = combined["volume"]
                # Cache for future runs
                self.data_engine.cache.save_ohlcv(binance_symbol, interval, combined)
            else:
                logger.warning("No Binance data for %s in range", coin)

        if not close_frames:
            return pd.DataFrame()

        prices = pd.DataFrame(close_frames).sort_index().ffill()
        return prices

    def _compute_nav(self, prices: pd.Series) -> float:
        """Compute portfolio NAV from current holdings + cash."""
        holdings_value = 0.0
        for coin, qty in self.holdings.items():
            if coin in prices.index:
                holdings_value += qty * prices[coin]
        return self.cash + holdings_value

    def _execute_rebalance(self, target_weights: pd.Series, prices: pd.Series,
                           nav: float, timestamp: pd.Timestamp) -> int:
        """Simulate rebalance execution at close prices.

        Returns number of trades executed.
        """
        trades = 0
        sells_first = []
        buys_second = []

        for coin in set(list(self.holdings.keys()) + list(target_weights.index)):
            price = prices.get(coin, 0)
            if price <= 0:
                continue

            current_qty = self.holdings.get(coin, 0)
            current_value = current_qty * price
            current_weight = current_value / nav if nav > 0 else 0

            target_weight = target_weights.get(coin, 0)
            target_value = target_weight * nav
            target_qty = target_value / price

            diff_qty = target_qty - current_qty
            diff_value = abs(diff_qty * price)

            # Skip tiny trades
            if diff_value < cfg.REBALANCE.min_trade_value_usd:
                continue
            weight_change = abs(target_weight - current_weight)
            if weight_change < cfg.REBALANCE.min_weight_change:
                continue

            if diff_qty < 0:
                sells_first.append((coin, diff_qty, price))
            else:
                buys_second.append((coin, diff_qty, price))

        # Execute sells first (free up cash)
        for coin, diff_qty, price in sells_first:
            sell_qty = min(abs(diff_qty), self.holdings.get(coin, 0))
            if sell_qty <= 0:
                continue
            proceeds = sell_qty * price
            commission = proceeds * self.COMMISSION_RATE
            self.cash += proceeds - commission
            self.holdings[coin] = self.holdings.get(coin, 0) - sell_qty
            if self.holdings[coin] < 1e-10:
                self.holdings.pop(coin, None)
                self.entry_prices.pop(coin, None)
            trades += 1
            self.trade_log.append({
                "timestamp": str(timestamp),
                "pair": f"{coin}/USD", "side": "SELL",
                "quantity": sell_qty, "price": price,
                "value": proceeds, "commission": commission,
            })

        # Execute buys
        for coin, diff_qty, price in buys_second:
            buy_value = diff_qty * price
            commission = buy_value * self.COMMISSION_RATE
            total_cost = buy_value + commission

            # Partial fill if not enough cash
            if total_cost > self.cash:
                if self.cash < cfg.REBALANCE.min_trade_value_usd:
                    continue
                buy_value = self.cash / (1 + self.COMMISSION_RATE)
                diff_qty = buy_value / price
                commission = buy_value * self.COMMISSION_RATE
                total_cost = buy_value + commission

            self.cash -= total_cost
            prev_qty = self.holdings.get(coin, 0)
            self.holdings[coin] = prev_qty + diff_qty
            if coin not in self.entry_prices:
                self.entry_prices[coin] = price
            trades += 1
            self.trade_log.append({
                "timestamp": str(timestamp),
                "pair": f"{coin}/USD", "side": "BUY",
                "quantity": diff_qty, "price": price,
                "value": buy_value, "commission": commission,
            })

        return trades

    def run(self) -> Dict:
        """Run the backtest.

        Returns:
            Dict with summary statistics and file paths.
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("=" * 60)
        logger.info("BACKTEST: %s -> %s (rebalance every %dh)",
                     self.start.date(), self.end.date(), self.rebalance_hours)
        logger.info("Initial capital: $%.2f | Universe: %d coins",
                     self.initial_capital, len(cfg.COINS))

        # 1. Fetch all historical data upfront
        warmup_days = cfg.SIGNALS.regime_lookback_hours // 24 + 10
        fetch_start = self.start - timedelta(days=warmup_days)
        fetch_end = self.end + timedelta(days=1)
        total_days = (fetch_end - fetch_start).days
        logger.info("Fetching %d days of historical data (%s to %s)...",
                     total_days, fetch_start.date(), fetch_end.date())

        close_prices = self._fetch_historical_range(
            cfg.COINS, fetch_start, fetch_end, interval="1h",
        )
        if close_prices.empty:
            logger.error("No historical data available for backtest")
            return {"status": "FAILED", "reason": "no_data"}
        # Compute returns per-column (don't dropna across all columns,
        # which would truncate to the newest asset's first row)
        returns = np.log(close_prices / close_prices.shift(1))
        # Drop only the first row (NaN from shift), keep per-column NaNs
        returns = returns.iloc[1:]

        if close_prices.empty or returns.empty:
            logger.error("No historical data available for backtest")
            return {"status": "FAILED", "reason": "no_data"}

        # Only use coins that have data
        available_coins = [c for c in cfg.COINS if c in close_prices.columns]
        if not available_coins:
            logger.error("No coins with data available")
            return {"status": "FAILED", "reason": "no_coins_with_data"}

        close_prices = close_prices[available_coins]
        returns = returns[[c for c in available_coins if c in returns.columns]]

        # Warn if effective start date differs from requested start
        bt_data = close_prices.loc[self.start:self.end].dropna(how="all")
        if not bt_data.empty:
            effective_start = bt_data.index[0]
            if effective_start > self.start + pd.Timedelta(hours=24):
                logger.warning("Effective trading start %s is later than requested %s "
                               "(earliest asset data begins at %s)",
                               effective_start.date(), self.start.date(),
                               effective_start.date())

        logger.info("Data loaded: %d rows, %d coins: %s",
                     len(close_prices), len(available_coins), available_coins)

        # 2. Walk forward through time
        # Get timestamps within the backtest window
        bt_timestamps = close_prices.loc[self.start:self.end].index
        if bt_timestamps.empty:
            logger.error("No data within backtest window %s to %s", self.start, self.end)
            return {"status": "FAILED", "reason": "no_data_in_window"}

        rebalance_count = 0
        total_trades = 0
        last_rebalance_ts = None
        continuous_mode = self.continuous_mode

        for i, ts in enumerate(bt_timestamps):
            # Compute NAV at every timestamp
            current_prices = close_prices.loc[ts].dropna()
            nav = self._compute_nav(current_prices)
            self.risk_mgr.update_nav(nav)
            self.nav_series.append((ts, nav))

            # Check if it's time to rebalance.
            should_rebalance = continuous_mode
            if not continuous_mode:
                if last_rebalance_ts is None:
                    should_rebalance = True
                else:
                    hours_since = (ts - last_rebalance_ts).total_seconds() / 3600
                    if hours_since >= self.rebalance_hours:
                        should_rebalance = True

            if not should_rebalance:
                continue

            # Get lookback window for signals
            lookback_data = close_prices.loc[:ts]
            lookback_returns = returns.loc[:ts]

            if len(lookback_data) < cfg.SIGNALS.regime_lookback_hours // 2:
                continue  # not enough warmup data yet

            # Build volume data for this lookback window
            vol_frames = {}
            for coin in available_coins:
                if coin in self._volume_cache:
                    v = self._volume_cache[coin]
                    v_window = v.loc[:ts]
                    if not v_window.empty:
                        vol_frames[coin] = v_window
            volumes = pd.DataFrame(vol_frames) if vol_frames else None

            # Compute alpha
            alpha_scores = self.alpha_engine.compute_alpha(
                lookback_data, lookback_returns, volumes=volumes)

            # Check risk
            regime = self.alpha_engine.current_regime
            if self.risk_mgr.check_correlation_spike(lookback_returns):
                regime = 2  # force bear

            # Expire elapsed cooldown windows.
            self._refresh_cooldowns(ts)

            # Check position stops — ATR-based dynamic stops
            # (pass lookback_returns so RiskManager can compute per-asset vol)
            stop_pct = cfg.RISK.position_stop_loss_pct  # fallback only
            check_prices = {
                coin: current_prices.get(coin, 0)
                for coin in self.holdings
                if not self._is_on_cooldown(coin, ts)
            }
            stopped = self.risk_mgr.check_position_stops(
                check_prices, self.entry_prices, stop_pct_override=stop_pct,
                returns=lookback_returns)
            for coin in stopped:
                if coin in alpha_scores.index:
                    alpha_scores[coin] = -10.0
                self._set_cooldown(coin, ts)
                # Reset HWM so re-entry starts fresh
                self.risk_mgr.reset_position_hwm(coin)

            # Prevent re-entry for coins on cooldown
            for coin in list(self._stop_cooldown):
                if coin in alpha_scores.index:
                    alpha_scores[coin] = min(alpha_scores[coin], -1.0)

            self._apply_outlier_overrides(
                alpha_scores=alpha_scores,
                signal_scores=self.alpha_engine.last_signal_scores,
                returns=lookback_returns,
                now_ts=ts,
            )

            # Optimize
            target_weights, diagnostics = self.optimizer.optimize(
                alpha_scores=alpha_scores,
                returns=lookback_returns,
                regime=regime,
                current_drawdown=self.risk_mgr.current_drawdown,
            )

            if target_weights.empty:
                continue

            # Enforce turnover cap (matching live mode)
            proposed_turnover = 0.0
            for coin in set(list(self.holdings.keys()) + list(target_weights.index)):
                price = current_prices.get(coin, 0)
                if price <= 0 or nav <= 0:
                    continue
                current_weight = (self.holdings.get(coin, 0) * price) / nav
                target_weight = target_weights.get(coin, 0)
                proposed_turnover += abs(target_weight - current_weight)
            proposed_turnover /= 2  # two-sided -> one-sided

            if proposed_turnover > cfg.REBALANCE.max_turnover_pct:
                scale = cfg.REBALANCE.max_turnover_pct / proposed_turnover
                # Scale target weights toward current weights
                for coin in target_weights.index:
                    price = current_prices.get(coin, 0)
                    current_weight = (self.holdings.get(coin, 0) * price) / nav if (price > 0 and nav > 0) else 0
                    target_weights[coin] = current_weight + scale * (target_weights[coin] - current_weight)
                target_weights = target_weights.clip(lower=0)

            # Execute rebalance
            n_trades = self._execute_rebalance(target_weights, current_prices, nav, ts)
            total_trades += n_trades
            rebalance_count += 1
            last_rebalance_ts = ts

            # Log snapshot
            regime_names = {0: "BULL", 1: "NEUTRAL", 2: "BEAR"}
            self.portfolio_log.append({
                "timestamp": str(ts),
                "nav_usd": nav,
                "cash_usd": self.cash,
                "n_positions": len(self.holdings),
                "invested_pct": 1 - (self.cash / nav) if nav > 0 else 0,
                "regime": regime_names.get(regime, "UNKNOWN"),
                "drawdown": self.risk_mgr.current_drawdown,
                "max_drawdown": self.risk_mgr.max_drawdown,
                "n_trades": n_trades,
            })

            if rebalance_count % 50 == 0:
                logger.info("  [%s] Rebalance #%d | NAV: $%.2f | DD: %.1f%% | Trades: %d",
                             ts.strftime("%Y-%m-%d %H:%M"), rebalance_count,
                             nav, self.risk_mgr.current_drawdown * 100, total_trades)

        # 3. Compute final metrics
        if not self.nav_series:
            logger.error("No NAV data generated — backtest produced no results")
            return {"status": "FAILED", "reason": "no_nav_data"}

        final_nav = self.nav_series[-1][1]
        total_return = (final_nav - self.initial_capital) / self.initial_capital

        nav_values = np.array([n for _, n in self.nav_series])
        nav_returns = np.diff(nav_values) / nav_values[:-1]
        nav_returns = nav_returns[np.isfinite(nav_returns)]

        sortino = self.risk_mgr.compute_sortino_ratio(nav_returns)
        sharpe = self.risk_mgr.compute_sharpe_ratio(nav_returns)
        calmar = self.risk_mgr.compute_calmar_ratio(nav_returns)
        max_dd = self.risk_mgr.max_drawdown

        # Hackathon composite score
        composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

        # Total commission paid
        total_commission = sum(t.get("commission", 0) for t in self.trade_log)

        summary = {
            "status": "OK",
            "run_id": run_id,
            "start": str(self.start.date()),
            "end": str(self.end.date()),
            "rebalance_hours": self.rebalance_hours,
            "adaptive": self.alpha_engine.adaptive,
            "initial_capital": self.initial_capital,
            "final_nav": round(final_nav, 2),
            "total_return_pct": round(total_return * 100, 2),
            "sortino": round(sortino, 4),
            "sharpe": round(sharpe, 4),
            "calmar": round(calmar, 4),
            "composite_score": round(composite, 4),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "rebalance_count": rebalance_count,
            "total_trades": total_trades,
            "total_commission": round(total_commission, 2),
            "n_coins": len(available_coins),
        }

        # Save adaptive diagnostics if enabled
        if self.alpha_engine.adaptive:
            summary["adaptive_diagnostics"] = self.alpha_engine.adaptive_diagnostics

        # 4. Save results
        out_dir = os.path.join(cfg.CSV_DIR, f"backtest_{run_id}")
        os.makedirs(out_dir, exist_ok=True)

        # NAV series
        nav_df = pd.DataFrame(self.nav_series, columns=["timestamp", "nav_usd"])
        nav_df.to_csv(os.path.join(out_dir, "nav_series.csv"), index=False)

        # Trade log
        if self.trade_log:
            pd.DataFrame(self.trade_log).to_csv(
                os.path.join(out_dir, "trade_log.csv"), index=False)

        # Portfolio snapshots
        if self.portfolio_log:
            pd.DataFrame(self.portfolio_log).to_csv(
                os.path.join(out_dir, "portfolio_log.csv"), index=False)

        # Summary
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        summary["output_dir"] = out_dir

        # Print results
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info("Period:           %s -> %s", summary["start"], summary["end"])
        logger.info("Initial capital:  $%.2f", self.initial_capital)
        logger.info("Final NAV:        $%.2f", final_nav)
        logger.info("Total return:     %.2f%%", total_return * 100)
        logger.info("Max drawdown:     %.2f%%", max_dd * 100)
        logger.info("-" * 40)
        logger.info("Sortino ratio:    %.4f", sortino)
        logger.info("Sharpe ratio:     %.4f", sharpe)
        logger.info("Calmar ratio:     %.4f", calmar)
        logger.info("COMPOSITE SCORE:  %.4f  (0.4×Sortino + 0.3×Sharpe + 0.3×Calmar)", composite)
        logger.info("-" * 40)
        logger.info("Rebalances:       %d", rebalance_count)
        logger.info("Total trades:     %d", total_trades)
        logger.info("Commission paid:  $%.2f", total_commission)
        logger.info("Output:           %s", out_dir)

        return summary


# ── CLI ──

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Crypto Trading Bot for Roostoo")
    parser.add_argument("--once", action="store_true", help="Run single rebalance cycle")
    parser.add_argument("--status", action="store_true", help="Print portfolio status")
    parser.add_argument("--test", action="store_true", help="Test API connectivity")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--start", type=str, default=None,
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=None,
                        help="Initial capital for backtest (USD)")
    parser.add_argument("--adaptive", action="store_true", default=True,
                        help="Enable adaptive signal weighting in backtest (default: on)")
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false",
                        help="Disable adaptive signal weighting in backtest")
    parser.add_argument("--rebalance-hours", type=int, default=None,
                        help="Override rebalance frequency (hours)")
    args = parser.parse_args()

    if args.backtest:
        if not args.start or not args.end:
            parser.error("--backtest requires --start and --end dates")
        engine = BacktestEngine(
            start=args.start,
            end=args.end,
            initial_capital=args.capital,
            adaptive=args.adaptive,
            rebalance_hours=args.rebalance_hours,
        )
        result = engine.run()
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0 if result.get("status") == "OK" else 1)

    bot = TradingBot()

    if args.test:
        success = bot.test_connectivity()
        sys.exit(0 if success else 1)

    if args.status:
        status = bot.get_portfolio_status()
        print(json.dumps(status, indent=2, default=str))
        sys.exit(0)

    if args.once:
        result = bot.run_rebalance_cycle()
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)

    # Default: continuous loop
    bot.run_loop()


if __name__ == "__main__":
    main()
