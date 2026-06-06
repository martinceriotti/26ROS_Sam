# Dashboard Guide — Student

## Logging In

Your instructor will provide your team's email and password. Go to the dashboard URL and log in. You land on the **Upload Predictions** page.

If you forgot your password, contact your instructor to reset it.

---

## Uploading Predictions

### Competition Submissions

1. Go to **Upload Predictions** → **Competition (Test Set)** tab.
2. Select the round you are submitting for (only open rounds are shown).
3. Upload a CSV with exactly two columns: `zpid` and `predicted_price` (5,038 rows matching the test set). Prices should be in dollars, not log-transformed.
4. The dashboard validates your file: correct columns, all zpids present, no missing or negative values.
5. Click **Submit Predictions**.

You can re-upload for the same round as many times as you want while the round is open. Once the organizer locks a round, submissions close.

If you only submit for round 1, your round 1 predictions are automatically used (forward-filled) for any later rounds you haven't submitted for.

### Practice Submissions (OOF)

1. Go to **Upload Predictions** → **Practice (Train Set OOF)** tab.
2. Upload out-of-fold predictions on the **training set** (11,840 rows, same `zpid, predicted_price` format).
3. Give each upload a label (e.g., "lgbm_baseline", "multimodal_v2") so you can tell them apart.

Practice submissions are used in the Practice Simulation page (see below). They do not affect the competition.

---

## Competition Results

Results appear after the organizer runs a simulation and releases the round. If a round has been evaluated but not released, you will see a message saying results are pending.

### Leaderboard

A summary table showing all teams, with your team's row highlighted. You can filter by round or view all rounds combined. The columns are:


| Metric                     | What it means                                                                                                                                                                          |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mean ROI (%)**           | Average return on investment across all simulated scenarios. Positive = your model makes money on average. This is the primary ranking metric.                                         |
| **Median ROI (%)**         | The middle scenario — half your outcomes are better, half are worse. Less sensitive to extreme outliers than the mean.                                                                 |
| **Std ROI (%)**            | How much your returns vary from scenario to scenario. Lower is more consistent. Two teams with the same mean ROI but different std have very different risk profiles.                  |
| **VaR 5% (%)**             | Value at Risk — the ROI at the 5th percentile. "In 95% of scenarios, you do at least this well." A measure of your worst realistic outcome.                                            |
| **CVaR 5% (%)**            | Conditional Value at Risk — the average ROI in your worst 5% of scenarios. How bad things get when they go bad. Always worse (lower) than VaR.                                         |
| **Prob Positive (%)**      | The percentage of scenarios where you made money (ROI > 0). A model with 80% prob positive turns a profit 4 out of 5 times.                                                            |
| **Mean Properties Bought** | How many properties your model purchases on average per scenario (out of 250 available). A model that buys very few is too conservative; one that buys too many may be overpredicting. |
| **Hit Rate (%)**           | Of the properties you bought, what fraction were profitable (true value > cost paid). High hit rate = your model is good at picking winners.                                           |
| **Sharpe**                 | Mean ROI divided by Std ROI. Measures risk-adjusted return — how much return you get per unit of volatility. Higher is better. A Sharpe above 1.0 is strong.                           |


### ROI Distribution

- Box plot comparing all teams' ROI distributions.
- Histogram overlay — select specific teams to compare their return distributions.

### Drill-Down (5 Representative Rounds)

The system selects 5 representative simulation rounds at percentiles P0 (your worst), P25, P50 (median), P75, and P100 (your best). These are fixed — you see the same 5 every time you log in.

For each of these 5 rounds, you see a table of the 250 properties that appeared on the market:


| What you see          | When                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------- |
| Your prediction       | Always                                                                                   |
| Decision reason       | Always — "Bid", "Skip: below threshold", or "Skip: no budget"                            |
| Asking price          | Only if you bid on the property                                                          |
| Auction details       | If you bid and the property was contested — all bids with team names, winner, price paid |
| True value and profit | Only if you won the property (bought it)                                                 |


Properties you skipped show your prediction and the reason you skipped, but not the asking price or true value.

### Property Map

An interactive map shows the 250 properties in the selected drill-down round, colored by outcome:

- **Gray** — Skip (you didn't bid)
- **Dark yellow** — Auction Lost (you bid but lost)
- **Dark green** — Profit (you bought and made money)
- **Dark red** — Loss (you bought and lost money)

Click a marker to see the property's listing photos below the map.

---

## Practice Simulation

1. Go to **Practice (OOF)**.
2. Select 2 or more of your uploaded practice prediction sets.
3. Optionally adjust simulation parameters (transaction cost, market noise, etc.).
4. Click **Run Practice Simulation**.

The simulation runs on the **training set** (where true prices are known), so you get full visibility: true values are shown for all properties, not just ones you bought. Use this to understand how the simulation works and how different models compare before submitting to the competition.

Results are displayed in the same format as competition results: summary table, box plot, and drill-down with full property details.

---

## My Submissions

A history of all your uploads — competition and practice — with timestamps and status (active or replaced).

---

## Tips

- **Start with practice** — upload two different OOF predictions and run a practice simulation to understand the mechanics before your first competition submission.
- **Each competition round uses different properties** — you cannot improve by memorizing feedback from earlier rounds. Focus on improving your model generally.
- **Forward-fill works in your favor** — if you're happy with your current model, you don't need to re-submit every round.
- **Auction dynamics matter** — if multiple teams predict similar values, you'll compete in auctions. Overpredicting wins auctions but overpays. Accuracy wins.

