# Options Analysis Toolkit

A comprehensive project on option pricing models, volatility analysis, and hedging strategies, including implementations of the Black-Scholes, Binomial Tree, and Monte Carlo models. This toolkit is ideal for exploring both theoretical and practical applications in options pricing and volatility-based trading.

## Project Features

1. **Option Pricing Models**: Includes implementations of major pricing models:
   - **Black-Scholes Model**: Closed-form solution for European call and put options.
   - **Binomial Tree Model**: Pricing method suitable for American options with flexibility for early exercise.
   - **Monte Carlo Simulation**: Useful for path-dependent options and to validate closed-form solutions.

2. **Implied Volatility & Volatility Smile Analysis**:
   - **Implied Volatility Calculation**: Find the market-implied volatility for a given option price using the Black-Scholes model and Newton-Raphson method.
   - **Volatility Smile**: Calculate and visualize how implied volatility changes with strike prices, highlighting market sentiment.

3. **Delta-Neutral Hedging**:
   - Use Delta-neutral hedging to minimize directional risk, isolating exposure to volatility changes rather than underlying price movements.

4. **Volatility-Based Trading Strategy**:
   - A mean-reversion strategy based on differences between historical and implied volatility. Takes advantage of potential mean-reversion in volatility through Delta-neutral, long volatility positions.
  
## Setup

To run this project, you'll need Python and the following libraries:

- `numpy`: For numerical calculations
- `scipy`: For statistical functions and root-finding
- `matplotlib`: For visualizations
- `yfinance` (optional): To fetch historical stock data



##Code Overview
Key Functions (in functions.py)

    black_scholes_call(S, K, T, r, sigma): Calculates the Black-Scholes price of a European call option.
    black_scholes_put(S, K, T, r, sigma): Calculates the Black-Scholes price of a European put option.
    binomial_tree_option(S, K, T, r, sigma, steps, option_type): Prices an option using the Binomial Tree method.
    monte_carlo_option_price(S, K, T, r, sigma, n_simulations): Prices an option using Monte Carlo simulation.
    implied_volatility_call(S, K, T, r, market_price): Uses the Newton-Raphson method to calculate implied volatility for a call option.
    delta_call(S, K, T, r, sigma): Computes the Delta of a European call option.
    delta_put(S, K, T, r, sigma): Computes the Delta of a European put option.

##Strategy Workflow

The volatility-based trading strategy is outlined in the notebook as follows:

    Setup: Define parameters, including stock price, strike price, maturity, and implied volatility.
    Create Straddle: Buy both a call and a put at the ATM strike price to capture volatility moves without directional risk.
    Delta-Neutral Hedge: Calculate initial Delta and short the underlying asset to neutralize directional exposure.
    Daily Monitoring: Adjust Delta hedges as needed over a 30-day horizon.
    Exit Conditions: Exit when implied volatility reverts to the historical level or the time horizon is reached.

##Sample Output

    Initial Cost of Straddle (Call + Put): 16.02
    Initial Delta Hedge: Short 0.27 shares to remain Delta-neutral.
    Delta Adjustments: Minor adjustments made only when Delta changed significantly.
    Exit Condition: Position exited after 30 days or when implied volatility reached the historical target.

##Visualization

The project includes several visualizations:

    Volatility Smile: A plot showing implied volatility across different strike prices.
    Historical vs. Implied Volatility Comparison: A chart illustrating the difference between market-implied and historical volatilities, highlighting potential trading signals.

##Project Summary

This toolkit provides a versatile approach to options analysis, combining theoretical pricing models with practical trading strategies. It's designed for anyone interested in understanding options pricing, managing risk through Delta-neutral hedging, and analyzing market sentiment via implied volatility.
Future Enhancements

##Possible improvements for this project include:

    Adding a volatility surface to analyze implied volatility across different maturities.
    Enhancing the backtesting framework to assess the volatility-based strategy over historical data.
    Expanding the toolkit to support more exotic options like Asian or barrier options.


Clone the Repository:
```bash
git clone https://github.com/yourusername/options-analysis-toolkit.git
cd options-analysis-toolkit

Open the Jupyter Notebook:

Launch Jupyter and open notebook.ipynb to follow the analysis step-by-step.

Run the Notebook Sections:

    Each section of the notebook focuses on a different component of the analysis:
        Option Pricing Models: Calculate option prices using Black-Scholes, Binomial Tree, and Monte Carlo models.
        Implied Volatility and Volatility Smile: Compute implied volatility across strike prices and plot the volatility smile.
        Delta-Neutral Hedging: Implement Delta-neutral hedging adjustments over time.
        Volatility-Based Strategy: Set up a mean-reversion strategy based on implied vs. historical volatility.
