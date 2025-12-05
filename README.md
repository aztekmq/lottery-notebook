## Self-Contained Lottery Notebook:

1. Lets you plug in live jackpots and **see EV curves** for Powerball & Mega Millions.
2. Adds a **MegaMillionsGame** class with the built-in 2×–10× multiplier distribution, using the official new prize matrix and probabilities. ([Maryland Lottery][1])

You can paste each “Cell” into a Jupyter/Colab notebook (or just run it as a plain `.py` file).

---

## 🧮 Cell 1 – Imports & basic data structures

```python
# Cell 1: imports & core data structures

from dataclasses import dataclass
from typing import List, Dict
import random
import math

import matplotlib.pyplot as plt  # for EV curves


@dataclass
class PrizeTier:
    """
    A non-jackpot prize tier in a lottery.

    Attributes
    ----------
    name : str
        Human-readable name for this tier.
    prize : float
        Base prize amount for this tier (before any multipliers).
    odds : float
        "1 in odds" probability; e.g., odds=38_859 means P = 1 / 38_859.
    """
    name: str
    prize: float
    odds: float


@dataclass
class LotteryGame:
    """
    Generic lottery game without built-in multipliers.
    """
    name: str
    ticket_cost: float
    jackpot_odds: float
    non_jackpot_tiers: List[PrizeTier]

    def tier_probabilities(self) -> Dict[str, float]:
        """
        Convert odds for jackpot + other tiers into probabilities
        and add a 'no_prize' bucket so everything sums to 1.
        """
        probs: Dict[str, float] = {}

        p_jackpot = 1.0 / self.jackpot_odds
        probs["jackpot"] = p_jackpot

        total_non_jackpot = 0.0
        for tier in self.non_jackpot_tiers:
            p = 1.0 / tier.odds
            probs[tier.name] = p
            total_non_jackpot += p

        p_no_prize = max(0.0, 1.0 - (p_jackpot + total_non_jackpot))
        probs["no_prize"] = p_no_prize

        # Normalize in case odds were rounded
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total

        return probs

    def expected_value(self, jackpot_cash_value: float) -> float:
        """
        Expected value per ticket given a jackpot *cash* value (not annuity),
        ignoring taxes and jackpot splitting.
        """
        probs = self.tier_probabilities()
        ev = 0.0

        # Jackpot component
        ev += probs["jackpot"] * jackpot_cash_value

        # Non-jackpot fixed prizes
        tier_by_name = {t.name: t for t in self.non_jackpot_tiers}
        for name, p in probs.items():
            if name in tier_by_name:
                ev += p * tier_by_name[name].prize

        return ev - self.ticket_cost
```

---

## 🎫 Cell 2 – Powerball configuration (no multiplier)

Uses current Powerball odds/prizes (no Power Play). ([Wisconsin Lottery][2])

```python
# Cell 2: Powerball configuration (without Power Play)

def make_powerball_game() -> LotteryGame:
    """
    Approximate modern Powerball (no Power Play, no Double Play).
    Odds & prize amounts from official odds tables.
    """
    non_jackpot_tiers = [
        PrizeTier("5_only",   1_000_000, 11_688_053.52),  # Match 5, no Powerball
        PrizeTier("4_PB",        50_000,    913_129.18),  # 4 + PB
        PrizeTier("4_only",          100,     36_525.17), # 4 only
        PrizeTier("3_PB",            100,     14_494.11), # 3 + PB
        PrizeTier("3_only",            7,        579.76), # 3 only
        PrizeTier("2_PB",              7,        701.33), # 2 + PB
        PrizeTier("1_PB",              4,         91.98), # 1 + PB
        PrizeTier("0_PB",              4,         38.32), # 0 + PB
    ]

    return LotteryGame(
        name="Powerball",
        ticket_cost=2.0,
        jackpot_odds=292_201_338.0,
        non_jackpot_tiers=non_jackpot_tiers,
    )

# Quick sanity check:
pb = make_powerball_game()
example_cash_jackpot = 383_500_000  # e.g., $383.5M cash (roughly an $800M+ advertised jackpot)
print("Powerball EV per ticket (cash jackpot ${:,}): ${:.4f}"
      .format(example_cash_jackpot, pb.expected_value(example_cash_jackpot)))
```

---

## 💥 Cell 3 – Mega Millions class with built-in multiplier

Here we encode the **new 2025 Mega Millions matrix** and the **2×–10× multiplier distribution**, based on the Maryland Lottery prize-probability table. ([Maryland Lottery][1])

```python
# Cell 3: Mega Millions with built-in multiplier

@dataclass
class MegaMillionsGame(LotteryGame):
    """
    Mega Millions with an embedded random multiplier (2x–10x) on
    *non-jackpot* prizes.

    multiplier_probs:
        Dict[multiplier -> probability]
        e.g., {2: p_2x, 3: p_3x, 4: p_4x, 5: p_5x, 10: p_10x}
    """
    multiplier_probs: Dict[int, float]

    def expected_multiplier(self) -> float:
        """Return E[multiplier]."""
        return sum(m * p for m, p in self.multiplier_probs.items())

    def expected_value(self, jackpot_cash_value: float) -> float:
        """
        EV per $5 ticket for the new Mega Millions game (post–Apr 8, 2025),
        given a jackpot *cash* value.
        All non-jackpot prizes are base_prize * random multiplier.
        """
        probs = self.tier_probabilities()
        ev = 0.0

        # Jackpot term
        ev += probs["jackpot"] * jackpot_cash_value

        # Non-jackpot prizes get multiplied by random multiplier
        tier_by_name = {t.name: t for t in self.non_jackpot_tiers}
        E_mult = self.expected_multiplier()

        for name, p in probs.items():
            if name in tier_by_name:
                base = tier_by_name[name].prize
                ev += p * base * E_mult

        return ev - self.ticket_cost


def make_megamillions_game() -> MegaMillionsGame:
    """
    Mega Millions new $5 format (effective April 8, 2025).

    Prize odds and base prizes are taken from the Maryland Lottery
    Mega Millions prize structure. :contentReference[oaicite:3]{index=3}
    Multiplier probabilities (2x–10x) are from the same page.
    """

    # Non-jackpot tiers: name, base prize, odds (1 in odds)
    # Match 5+MB (jackpot) handled separately via jackpot_odds.
    # Rows below correspond to:
    # 5 only, 4+MB, 4 only, 3+MB, 3 only, 2+MB, 1+MB, 0+MB.
    non_jackpot_tiers = [
        PrizeTier("5_only", 1_000_000, 12_629_232),  # 5, no Mega Ball
        PrizeTier("4_MB",      10_000,    893_761),  # 4 + MB
        PrizeTier("4_only",        500,     38_859), # 4 only
        PrizeTier("3_MB",          200,     13_965), # 3 + MB
        PrizeTier("3_only",         10,        607), # 3 only
        PrizeTier("2_MB",           10,        665), # 2 + MB
        PrizeTier("1_MB",            7,         86), # 1 + MB
        PrizeTier("0_MB",            5,         35), # 0 + MB
    ]

    # Multiplier distribution (approximate, from MD Lottery):
    #   P(2x) = 1 / 2.13
    #   P(3x) = 1 / 3.2
    #   P(4x) = 1 / 8
    #   P(5x) = 1 / 16
    #   P(10x) = 1 / 32  :contentReference[oaicite:4]{index=4}
    multiplier_probs = {
        2: 1.0 / 2.13,
        3: 1.0 / 3.2,
        4: 1.0 / 8.0,
        5: 1.0 / 16.0,
        10: 1.0 / 32.0,
    }
    # Normalize to exactly sum to 1
    total = sum(multiplier_probs.values())
    for m in multiplier_probs:
        multiplier_probs[m] /= total

    return MegaMillionsGame(
        name="Mega Millions",
        ticket_cost=5.0,
        jackpot_odds=290_472_336.0,  # new odds post-Apr 8 2025 :contentReference[oaicite:5]{index=5}
        non_jackpot_tiers=non_jackpot_tiers,
        multiplier_probs=multiplier_probs,
    )


# Quick sanity check:
mm = make_megamillions_game()
print("Mega Millions expected multiplier:", mm.expected_multiplier())

example_mm_cash_jackpot = 208_000_000  # e.g., cash value if advertised jackpot ~ $450M
print("Mega Millions EV per ticket (cash jackpot ${:,}): ${:.4f}"
      .format(example_mm_cash_jackpot, mm.expected_value(example_mm_cash_jackpot)))
```

---

## 📈 Cell 4 – EV curve helpers (for “live” jackpots)

This will let you plug in a range of jackpot **cash values** and see how EV changes.

> 💡 Note: Lottery sites usually show **annuity** jackpot. Cash value is typically ~50–60% of that; use the cash amount from official pages when you can.

```python
# Cell 4: EV curve utilities

def ev_curve(game: LotteryGame, jackpots_cash: List[float]) -> List[float]:
    """
    Compute EV per ticket for each jackpot cash value in jackpots_cash.
    """
    return [game.expected_value(J) for J in jackpots_cash]


def plot_ev_curve(
    game: LotteryGame,
    min_jackpot_cash: float,
    max_jackpot_cash: float,
    num_points: int = 50,
):
    """
    Plot EV per ticket vs. jackpot *cash* value for a given game.
    """
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    step = (max_jackpot_cash - min_jackpot_cash) / (num_points - 1)
    jackpots = [min_jackpot_cash + i * step for i in range(num_points)]
    evs = ev_curve(game, jackpots)

    plt.figure()
    plt.plot(jackpots, evs, label=f"{game.name} EV")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Jackpot cash value ($)")
    plt.ylabel("Expected value per ticket ($)")
    plt.title(f"{game.name}: EV vs. jackpot (cash value)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return jackpots, evs
```

---

## 🔢 Cell 5 – Example: plug in current jackpots

Adjust these inputs whenever you want; just re-run this cell.

```python
# Cell 5: Example usage with "live" jackpots

pb = make_powerball_game()
mm = make_megamillions_game()

# --- You update these two values manually based on current cash options ---
powerball_cash_jackpot_today = 383_500_000  # example
megamillions_cash_jackpot_today = 208_000_000  # example

print(f"Powerball EV today: ${pb.expected_value(powerball_cash_jackpot_today):.4f}")
print(f"Mega Millions EV today: ${mm.expected_value(megamillions_cash_jackpot_today):.4f}")

# EV curves over a range of cash jackpots
# (adjust ranges to match what's realistic for today)
plot_ev_curve(pb, min_jackpot_cash=100_000_000, max_jackpot_cash=1_000_000_000, num_points=50)
plot_ev_curve(mm, min_jackpot_cash=50_000_000, max_jackpot_cash=800_000_000, num_points=50)
```

If you’d like, next step I can add:

* A “syndicate simulator” cell that uses these exact `LotteryGame` objects.
* Or a small helper that converts *advertised* jackpot → *approx cash* automatically (using a configurable factor).

[1]: https://www.mdlottery.com/games/mega-millions/prize-structure/ "Prize Structure – Maryland Lottery"
[2]: https://wilottery.com/mega-millions-new-features?utm_source=chatgpt.com "Mega Millions New Features"
