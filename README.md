## 1. Streamlit app: Lottery EV Explorer

### 1.1. Create a project folder

Pick a directory, e.g.:

```bash
mkdir lottery_ev_app
cd lottery_ev_app
```

### 1.2. (Optional) Create and activate a virtual environment

**On Windows (PowerShell):**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**On macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3. Install dependencies

```bash
pip install streamlit matplotlib
```

(If you don’t have `pip` / `python3`, install Python 3.10+ first.)

### 1.4. Create `app.py` with this code

Copy/paste this entire block into a file called `app.py` in that folder:

```python
from dataclasses import dataclass
from typing import List, Dict
import random

import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Core data structures & models
# -----------------------------

@dataclass
class PrizeTier:
    name: str
    prize: float
    odds: float  # "1 in odds" (e.g. 38_859 => p = 1/38_859)


@dataclass
class LotteryGame:
    name: str
    ticket_cost: float
    jackpot_odds: float
    non_jackpot_tiers: List[PrizeTier]

    def tier_probabilities(self) -> Dict[str, float]:
        """
        Convert odds to probabilities (including 'no_prize') so everything sums to 1.
        """
        probs: Dict[str, float] = {}

        # Jackpot probability
        p_jackpot = 1.0 / self.jackpot_odds
        probs["jackpot"] = p_jackpot

        # Non-jackpot tiers
        total_non_jackpot = 0.0
        for tier in self.non_jackpot_tiers:
            p = 1.0 / tier.odds
            probs[tier.name] = p
            total_non_jackpot += p

        # "No prize" catch-all
        p_no_prize = max(0.0, 1.0 - (p_jackpot + total_non_jackpot))
        probs["no_prize"] = p_no_prize

        # Normalize (odds are approximate, so this tightens it)
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total

        return probs

    def expected_value(self, jackpot_cash_value: float) -> float:
        """
        EV per ticket given a *cash* jackpot (ignores taxes & jackpot-splitting):

        EV = sum_k p_k * payout_k - ticket_cost
        """
        probs = self.tier_probabilities()
        ev = 0.0

        # Jackpot term
        ev += probs["jackpot"] * jackpot_cash_value

        # Fixed non-jackpot prizes
        tier_by_name = {t.name: t for t in self.non_jackpot_tiers}
        for name, p in probs.items():
            if name in tier_by_name:
                ev += p * tier_by_name[name].prize

        return ev - self.ticket_cost


@dataclass
class MegaMillionsGame(LotteryGame):
    """
    Mega Millions variant with 2x–10x multiplier on non-jackpot prizes.
    """
    multiplier_probs: Dict[int, float]  # e.g. {2: p_2x, 3: p_3x, 4: p_4x, ...}

    def expected_multiplier(self) -> float:
        """E[multiplier] for non-jackpot prizes."""
        return sum(m * p for m, p in self.multiplier_probs.items())

    def expected_value(self, jackpot_cash_value: float) -> float:
        """
        Override EV: apply multiplier on non-jackpot prizes.
        """
        probs = self.tier_probabilities()
        ev = 0.0

        # Jackpot term
        ev += probs["jackpot"] * jackpot_cash_value

        # Multiplied non-jackpot prizes
        tier_by_name = {t.name: t for t in self.non_jackpot_tiers}
        E_mult = self.expected_multiplier()

        for name, p in probs.items():
            if name in tier_by_name:
                base = tier_by_name[name].prize
                ev += p * base * E_mult

        return ev - self.ticket_cost


# -----------------------------
# Game constructors
# -----------------------------

def make_powerball_game() -> LotteryGame:
    """
    Approximate modern Powerball (no Power Play, no Double Play).
    """
    non_jackpot_tiers = [
        PrizeTier("5_only",   1_000_000, 11_688_053.52),
        PrizeTier("4_PB",        50_000,    913_129.18),
        PrizeTier("4_only",          100,     36_525.17),
        PrizeTier("3_PB",            100,     14_494.11),
        PrizeTier("3_only",            7,        579.76),
        PrizeTier("2_PB",              7,        701.33),
        PrizeTier("1_PB",              4,         91.98),
        PrizeTier("0_PB",              4,         38.32),
    ]
    return LotteryGame(
        name="Powerball",
        ticket_cost=2.0,
        jackpot_odds=292_201_338.0,
        non_jackpot_tiers=non_jackpot_tiers,
    )


def make_megamillions_game() -> MegaMillionsGame:
    """
    New Mega Millions ($5, built-in 2x–10x multiplier).
    """
    non_jackpot_tiers = [
        PrizeTier("5_only", 1_000_000, 12_629_232),
        PrizeTier("4_MB",      10_000,    893_761),
        PrizeTier("4_only",        500,     38_859),
        PrizeTier("3_MB",          200,     13_965),
        PrizeTier("3_only",         10,        607),
        PrizeTier("2_MB",           10,        665),
        PrizeTier("1_MB",            7,         86),
        PrizeTier("0_MB",            5,         35),
    ]

    # Approx multiplier distribution P(2x..10x)
    multiplier_probs = {
        2: 1.0 / 2.13,
        3: 1.0 / 3.2,
        4: 1.0 / 8.0,
        5: 1.0 / 16.0,
        10: 1.0 / 32.0,
    }
    total = sum(multiplier_probs.values())
    for m in multiplier_probs:
        multiplier_probs[m] /= total

    return MegaMillionsGame(
        name="Mega Millions",
        ticket_cost=5.0,
        jackpot_odds=290_472_336.0,
        non_jackpot_tiers=non_jackpot_tiers,
        multiplier_probs=multiplier_probs,
    )


# -----------------------------
# Syndicate simulation
# -----------------------------

def simulate_syndicate(
    game: LotteryGame,
    jackpot_cash_value: float,
    tickets_per_drawing: int,
    num_drawings: int,
    rng_seed: int = 42,
):
    """
    Monte Carlo: simulate buying `tickets_per_drawing` tickets for `num_drawings`.
    Returns basic ROI stats.
    """
    random.seed(rng_seed)
    probs = game.tier_probabilities()

    outcome_names = list(probs.keys())
    cum_probs = []
    cumulative = 0.0
    for name in outcome_names:
        cumulative += probs[name]
        cum_probs.append(cumulative)

    tier_by_name = {t.name: t for t in game.non_jackpot_tiers}

    def sample_outcome() -> str:
        r = random.random()
        for name, cp in zip(outcome_names, cum_probs):
            if r <= cp:
                return name
        return outcome_names[-1]  # fallback

    total_spent_all = 0.0
    total_won_all = 0.0
    drawing_rois = []

    for _ in range(num_drawings):
        spent = tickets_per_drawing * game.ticket_cost
        won = 0.0

        for _ in range(tickets_per_drawing):
            outcome = sample_outcome()
            if outcome == "jackpot":
                won += jackpot_cash_value
            elif outcome in tier_by_name:
                won += tier_by_name[outcome].prize

        total_spent_all += spent
        total_won_all += won
        roi = (won - spent) / spent
        drawing_rois.append(roi)

    avg_roi = sum(drawing_rois) / len(drawing_rois)
    worst_roi = min(drawing_rois)
    best_roi = max(drawing_rois)

    return {
        "total_spent": total_spent_all,
        "total_won": total_won_all,
        "overall_roi": (total_won_all - total_spent_all) / total_spent_all,
        "avg_drawing_roi": avg_roi,
        "worst_drawing_roi": worst_roi,
        "best_drawing_roi": best_roi,
    }


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.title("Lottery EV Explorer")

    st.markdown(
        """
        Interactive explorer for **Powerball** and **Mega Millions**:

        - Compute **EV per ticket** at a given cash jackpot  
        - Plot **EV vs. jackpot** over a range  
        - Run a **syndicate Monte Carlo simulation**  

        All values are approximate and for educational use only.
        """
    )

    # ---- Sidebar: game & jackpot ----
    st.sidebar.header("Game & jackpot")

    game_name = st.sidebar.selectbox("Lottery game", ["Powerball", "Mega Millions"])

    if game_name == "Powerball":
        game = make_powerball_game()
        default_cash = 383_500_000.0
    else:
        game = make_megamillions_game()
        default_cash = 208_000_000.0

    jackpot_cash = st.sidebar.number_input(
        "Current jackpot cash value ($)",
        min_value=10_000_000.0,
        max_value=2_000_000_000.0,
        value=default_cash,
        step=10_000_000.0,
        format="%.0f",
    )

    # ---- Sidebar: EV curve settings ----
    st.sidebar.markdown("---")
    st.sidebar.header("EV curve settings")

    min_cash = st.sidebar.number_input(
        "Min jackpot cash for EV curve ($)",
        min_value=10_000_000.0,
        max_value=2_000_000_000.0,
        value=max(50_000_000.0, jackpot_cash / 2),
        step=10_000_000.0,
        format="%.0f",
    )
    max_cash = st.sidebar.number_input(
        "Max jackpot cash for EV curve ($)",
        min_value=min_cash + 10_000_000.0,
        max_value=3_000_000_000.0,
        value=max(jackpot_cash * 1.5, min_cash + 50_000_000.0),
        step=10_000_000.0,
        format="%.0f",
    )
    num_points = st.sidebar.slider("Points on EV curve", 10, 200, 50)

    # ---- Sidebar: syndicate settings ----
    st.sidebar.markdown("---")
    st.sidebar.header("Syndicate simulation")

    tickets_per_drawing = st.sidebar.number_input(
        "Tickets per drawing",
        min_value=1,
        max_value=1_000_000,
        value=10_000,
        step=1_000,
    )
    num_drawings = st.sidebar.number_input(
        "Number of drawings to simulate",
        min_value=1,
        max_value=10_000,
        value=1_000,
        step=100,
    )
    rng_seed = st.sidebar.number_input(
        "Random seed", min_value=0, max_value=999_999, value=42, step=1
    )

    # ---- Section 1: EV at current jackpot ----
    st.header("1. EV per ticket at current jackpot")

    ev_now = game.expected_value(jackpot_cash)
    st.metric(
        label=f"{game.name} EV per ticket",
        value=f"${ev_now:0.4f}",
    )

    st.caption(
        "EV is based on approximate odds/prizes and the *cash* jackpot value. "
        "It ignores taxes and jackpot-splitting. "
        "Negative EV means a statistical loss on average."
    )

    # ---- Section 2: EV vs. jackpot ----
    st.header("2. EV vs. jackpot cash value")

    xs = [
        min_cash + i * (max_cash - min_cash) / (num_points - 1)
        for i in range(num_points)
    ]
    ys = [game.expected_value(x) for x in xs]

    fig, ax = plt.subplots()
    ax.plot(xs, ys, label="EV per ticket")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Jackpot cash value ($)")
    ax.set_ylabel("EV per ticket ($)")
    ax.set_title(f"{game.name} EV vs. jackpot cash value")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    if isinstance(game, MegaMillionsGame):
        st.caption(
            f"Mega Millions expected multiplier on non-jackpot prizes: "
            f"{game.expected_multiplier():.3f}x"
        )

    # ---- Section 3: Syndicate simulation ----
    st.header("3. Syndicate simulation")

    run_sim = st.button("Run simulation")

    if run_sim:
        results = simulate_syndicate(
            game=game,
            jackpot_cash_value=jackpot_cash,
            tickets_per_drawing=int(tickets_per_drawing),
            num_drawings=int(num_drawings),
            rng_seed=int(rng_seed),
        )
        st.subheader("Simulation summary")
        st.write(
            f"Total spent: ${results['total_spent']:,.0f}\n\n"
            f"Total won: ${results['total_won']:,.0f}\n\n"
            f"Overall ROI: {results['overall_roi']*100:0.2f}%"
        )
        st.write(
            f"Average per-drawing ROI: {results['avg_drawing_roi']*100:0.2f}% "
            f"(best: {results['best_drawing_roi']*100:0.2f}%, "
            f"worst: {results['worst_drawing_roi']*100:0.2f}%)"
        )
        st.caption(
            "Even with many tickets and drawings, the average ROI tends to "
            "converge near the theoretical EV implied by the model."
        )

    st.markdown("---")
    st.markdown(
        "This tool is for educational purposes only and does not constitute "
        "financial or gambling advice."
    )


if __name__ == "__main__":
    main()
```

### 1.5. Run the Streamlit app

From the same folder:

```bash
streamlit run app.py

or from within WSL

streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

From your browser, open to http://localhost:8501` with:

* A sidebar to choose **Powerball / Mega Millions**, set jackpot, sliders, etc.
* EV at current jackpot
* An EV vs jackpot plot
* A button to run the syndicate simulation and see ROI stats
