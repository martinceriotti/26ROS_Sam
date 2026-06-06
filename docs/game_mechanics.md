# Investment Competition — Game Mechanics

## The Business

Miami's real estate market is volatile, fast-moving, and full of information asymmetry. Distressed sellers — facing foreclosure, relocation, divorce, or estate liquidation — routinely list properties below their true market value because they need cash quickly. For an investment firm with the right analytical edge, this creates a repeatable opportunity: identify genuinely underpriced properties, acquire them, and flip at true market value.

This is not hypothetical. Firms like Opendoor, Offerpad, and dozens of regional flippers operate exactly this model. The economics are tight — transaction costs eat into margins, competition from other buyers drives prices up, and a single bad purchase can wipe out the profit from several good ones. The firms that survive are the ones with the best pricing models.

In this competition, you are that firm.

---

## The Decisions Behind the Business

If you were actually running a property investment firm, your team would face a cascade of interconnected decisions on every deal:

**Valuation** — What is this property actually worth? Your entire strategy rests on this number. Get it wrong by 10% and a great deal becomes a money pit. Every other decision downstream — whether to bid, how much to offer, how much capital to risk — depends on your confidence in this estimate.

**Market reading** — How do you interpret the asking price? Is the seller truly distressed or just testing the market? How much noise is there between asking price and true value? A property listed at $400K might be worth $500K (a steal) or $350K (a trap). Your model needs to tell the difference.

**Deal selection** — Which properties are worth pursuing? You can't buy everything — capital is finite, and every dollar tied up in a mediocre deal is a dollar not available for a great one. You need a threshold: how much perceived upside justifies deploying capital?

**Bid strategy** — When competing against other buyers, how aggressively do you bid? Bid too high and you win every auction but overpay. Bid too low and you lose every contested deal to someone bolder. The optimal bid depends on your confidence in your valuation and your read on the competition.

**Portfolio construction** — How much capital do you concentrate in a single property? A $5M fund that puts $2M into one house is taking a massive idiosyncratic risk. Diversification protects against individual valuation errors but limits upside on your best picks.

**Capital deployment** — How much of your fund do you actually invest vs. hold in reserve? Being too aggressive means you run out of capital before the best deals appear. Being too conservative means your money sits idle earning nothing while opportunities pass.

Each of these decisions interacts with the others, and in the real world, experienced investors develop intuition for all of them over years of practice.

---

## Your Job in the Competition

We are going to simplify.

In this competition, your team is responsible for **one thing**: producing the most accurate property valuation model you can. You submit a CSV with your predicted market value for every property in the test set. That's your only deliverable.

All of the other decisions — deal selection, bid strategy, portfolio sizing, capital management — are handled by the simulation engine using fixed, transparent rules that are the same for every team. The simulation takes your predictions and plays the investment game on your behalf, making rational decisions based on how your predicted values compare to the market's asking prices.

**Why this simplification works**: in the real business, the quality of every downstream decision ultimately depends on the quality of the initial valuation. A firm with perfect price estimates can trivially optimize every other parameter. A firm with terrible estimates will lose money no matter how clever its bidding strategy. By fixing the strategy layer and letting teams compete purely on prediction accuracy, we isolate the variable that matters most — and the one that data science can actually improve.

**You will predict all test properties**: since you don't know in advance which properties will be in your round's pool, you need to provide an estimate on every test property. The simulation engine will then run a thousand different market scenarios on your round's pool to build a robust picture of your model's real investment performance.

---

## What's at Stake: Model Quality Drives Returns

The simulation translates prediction accuracy directly into investment returns. Here is what to expect with default competition parameters:

The table below shows actual simulation results from models built on this dataset, using default competition parameters. These are not hypothetical — they come from running 1,000 Monte Carlo simulations.


| Model                               | MAPE | Mean ROI |  Interpretation                                |
| ----------------------------------- | ---- | -------- |  --------------------------------------------- |
| Passive (buy nothing)               | —    | -4.0%    |  The floor — what happens if you sit on cash   |
| Image only                          | ~50% | -19%     |  Predictions too noisy to make any money       |
| Text only                           | ~43% | -10%     |  Some signal, but not nearly enough            |
| Tabular (minimal)                   | ~29% | -4%      |  Roughly the same as doing nothing             |
| Tabular (extended)                  | ~29% | -3%      |  Still losing, but getting closer to breakeven |
| Multimodal (tabular + image + text) | ~28% | +3%      |  Profitable — each modality adds real value    |


The gap between ~29% MAPE and ~28% MAPE looks small on a leaderboard, but in the simulation it's the difference between consistently losing money and turning a profit 4 out of 5 times. Every fraction of a percent matters when transaction costs and competition compress margins.

---

## Competition Mechanics in Detail

The rest of this document explains exactly how the simulation works. You don't need to memorize this to compete — your only job is still "make better predictions" — but understanding the mechanics helps you interpret your results and reason about why certain models outperform others.

### Simulation Overview

The competition is evaluated via **Monte Carlo simulation**: the investment game is played 1,000 times with different random market conditions each time. Your predictions are fixed across all simulations — only the property selection and market noise change. This produces a **distribution** of outcomes, not a single number, revealing how robust your model is.

Each competition round uses its own pool of ~1,260 properties (the test set is split into 4 non-overlapping pools). Within a round, the simulation runs 1,000 times. Each simulation consists of **4 internal investment rounds**. Each internal round, 250 properties are drawn from the pool (without replacement), giving ~1,000 properties per simulation out of ~1,260 available. Capital resets to $5M each internal round.

### Step-by-Step: What Happens Each Round

**Step 1 — Property selection** `[SIMULATION]`

The engine randomly draws 250 properties from the remaining pool for this round.

**Step 2 — Asking prices** `[SIMULATION]`

Each property gets a synthetic asking price:

```
asking_price = true_value × (1 + Normal(market_bias, market_noise))
```

Default: `market_bias = -0.07` (sellers underprice by 7% on average), `market_noise = 0.35` (high variance — some are great deals, others are traps).

**Step 3 — Buy/skip decisions** `[SIMULATION, using your predictions]`

Properties are presented sequentially. For each property, the simulation checks whether your team wants to buy. Two conditions must both be met:

1. **Signal check** — your model sees sufficient upside:
  ```
   prediction > asking_price × (1 + buy_threshold)
  ```
   Default threshold is 8%. If the asking price is $400K, your model must predict at least $432K to trigger a buy.
2. **Affordability check** — you can afford it:
  ```
   asking_price × (1 + transaction_cost) ≤ remaining_capital × max_position_pct
  ```
   You can't spend more than 25% of your remaining round capital on one property.

There is **no look-ahead**: the simulation cannot skip a mediocre deal to save capital for a better one later. This is realistic — in a real market you don't know what's coming next.

If both checks pass, your team's sealed bid is computed: `prediction × bid_fraction` (default 0.85).

**Step 4 — Resolution** `[SIMULATION]`

Each property resolves based on how many teams want it:

- **Nobody wants it** → the market complement portfolio buys at asking price
- **One team wants it** → uncontested purchase at asking price + transaction costs
- **Multiple teams want it** → second-price sealed-bid auction (see below)

**Step 5 — P&L** `[SIMULATION]`

For each property won:

```
cost = purchase_price × (1 + transaction_cost)
profit = true_value - cost
```

At the end of each round, idle capital is penalized:

```
penalty = idle_capital × (opportunity_cost_rate / n_rounds)
```

### Auctions

When 2+ teams want the same property, a **second-price sealed-bid (Vickrey) auction** decides the outcome:

1. Each team's bid was already computed: `prediction × bid_fraction`
2. **Highest bidder wins**
3. Winner pays: `max(asking_price, second_highest_bid)` plus transaction costs
4. Ties are broken by random coin flip

Why second-price? It's strategy-proof — your optimal bid is proportional to your true valuation, no incentive to bluff. You never pay your own bid, only the second-highest. Teams that overpredict win more auctions but systematically overpay. Teams that underpredict lose contests but avoid overpaying. **Accuracy wins.**

### ROI Calculation

**Teams:**

```
net = total_profit - total_opportunity_cost
total_capital = starting_capital × n_rounds
ROI = net / total_capital × 100
```

ROI is measured against total available capital (not just deployed capital). This prevents a team that buys almost nothing from showing a misleadingly high percentage return on its tiny investment.

**Market complement** (no capital constraint):

```
ROI = total_profit / total_invested × 100
```

### Win Condition

Each simulation produces an ROI for every team. The winner is the team with the highest ROI in that simulation. Across 1,000 simulations, your **win rate** is the fraction where you came out on top.

---

## Parameters Reference

All parameters are configurable in the dashboard. Defaults are calibrated so that a basic tabular model roughly breaks even, and each modality improvement produces visible but realistic gains.

### Competition Structure


| Parameter        | Default    | Range     | Purpose                                      |
| ---------------- | ---------- | --------- | -------------------------------------------- |
| Rounds           | 4          | 1–8       | Independent investment rounds per simulation |
| Properties/round | 250        | 10–500    | Properties offered each round                |
| Simulations      | 1,000      | 100–5,000 | Monte Carlo runs for statistical robustness  |
| Starting capital | $5,000,000 | $1M–$50M  | Fresh capital each round                     |


### Economics


| Parameter        | Default | Range     | Purpose                                           |
| ---------------- | ------- | --------- | ------------------------------------------------- |
| Transaction cost | 2%      | 1–10%     | Round-trip closing costs on every purchase        |
| Buy threshold    | 8%      | 0–20%     | Min perceived upside to trigger a buy             |
| Opportunity cost | 4%      | 0–8%      | Annual penalty on idle capital                    |
| Max position     | 25%     | 10–50%    | Max capital per single property (diversification) |
| Bid fraction     | 0.85    | 0.70–1.00 | Bid = prediction × this fraction in auctions      |


### Market


| Parameter       | Default | Range        | Purpose                                         |
| --------------- | ------- | ------------ | ----------------------------------------------- |
| Seller discount | -7%     | -15% to +10% | Systematic pricing bias (negative = distressed) |
| Market noise    | 35%     | 5–50%        | Random variation in asking prices               |


---

## Your Workflow

1. **Build a model** that predicts sale price for each test property
2. **Export a CSV** with columns `zpid, predicted_price` (5,038 rows)
3. **Log in** to the competition dashboard and upload your CSV for the current round
4. **Review results** once the organizer runs the simulation and releases results — ROI distribution, leaderboard, drill-down into representative scenarios, property map
5. **Iterate** — improve your model, resubmit for the next round

Each competition round evaluates on a separate pool of ~1,260 properties (the test set is split into non-overlapping pools). This means you cannot overfit to feedback from earlier rounds — the properties change each round.

