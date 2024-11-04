import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    
    Returns:
    float: Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European put option.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    
    Returns:
    float: Put option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def delta_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def theta_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return theta

def theta_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return theta

def rho_call(S, K, T, r, sigma):
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return K * T * np.exp(-r * T) * norm.cdf(d2)

def rho_put(S, K, T, r, sigma):
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def binomial_tree_call(S0, K, T, r, sigma, steps, american=False):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize stock price tree
    stock_prices = np.zeros((steps + 1, steps + 1))
    stock_prices[0, 0] = S0

    for i in range(1, steps + 1):
        stock_prices[i, 0] = stock_prices[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_prices[i, j] = stock_prices[i - 1, j - 1] * d

    # Initialize option value tree
    option_values = np.zeros((steps + 1, steps + 1))
    for j in range(steps + 1):
        option_values[steps, j] = max(0, stock_prices[steps, j] - K)  # payoff for call option

    # Work backwards
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold_value = np.exp(-r * dt) * (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1])
            if american:
                exercise_value = max(0, stock_prices[i, j] - K)
                option_values[i, j] = max(hold_value, exercise_value)
            else:
                option_values[i, j] = hold_value

    return option_values[0, 0]  # Return the option value at the root


def binomial_tree_put(S0, K, T, r, sigma, steps, american=False):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize stock price tree
    stock_prices = np.zeros((steps + 1, steps + 1))
    stock_prices[0, 0] = S0

    for i in range(1, steps + 1):
        stock_prices[i, 0] = stock_prices[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_prices[i, j] = stock_prices[i - 1, j - 1] * d

    # Initialize option value tree for put options
    option_values = np.zeros((steps + 1, steps + 1))
    for j in range(steps + 1):
        option_values[steps, j] = max(0, K - stock_prices[steps, j])  # payoff for put option

    # Work backwards through the tree
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold_value = np.exp(-r * dt) * (p * option_values[i + 1, j] + (1 - p) * option_values[i + 1, j + 1])
            if american:
                exercise_value = max(0, K - stock_prices[i, j])
                option_values[i, j] = max(hold_value, exercise_value)
            else:
                option_values[i, j] = hold_value

    return option_values[0, 0]  # Return the option value at the root

import numpy as np

def monte_carlo_option_price_antithetic(S0, K, T, r, sigma, n_simulations=10000, steps=100):
    dt = T / steps
    payoffs = []

    for _ in range(n_simulations // 2):  # Halve the number of loops for pairs
        S_t1, S_t2 = S0, S0  # Generate two paths, one regular and one antithetic
        for _ in range(steps):
            Z = np.random.normal()
            S_t1 = S_t1 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            S_t2 = S_t2 * np.exp((r - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt) * Z)
        
        # Average payoff from both paths
        payoffs.append(0.5 * (max(S_t1 - K, 0) + max(S_t2 - K, 0)))
    
    # Calculate the average payoff and discount to present
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def monte_carlo_asian_call(S0, K, T, r, sigma, n_simulations=10000, steps=100):
    dt = T / steps
    payoffs = []

    for _ in range(n_simulations):
        # Initialize the price path
        S_t = S0
        path = [S_t]
        
        # Simulate the path over the specified number of steps
        for _ in range(steps):
            Z = np.random.normal()
            S_t = S_t * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            path.append(S_t)
        
        # Calculate the average price over the path
        avg_price = np.mean(path)
        
        # Calculate the payoff for an Asian call option
        payoffs.append(max(avg_price - K, 0))
    
    # Calculate the average payoff and discount it back to the present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def monte_carlo_asian_put(S0, K, T, r, sigma, n_simulations=10000, steps=100):
    dt = T / steps
    payoffs = []

    for _ in range(n_simulations):
        # Initialize the price path
        S_t = S0
        path = [S_t]
        
        # Simulate the path over the specified number of steps
        for _ in range(steps):
            Z = np.random.normal()
            S_t = S_t * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            path.append(S_t)
        
        # Calculate the average price over the path
        avg_price = np.mean(path)
        
        # Calculate the payoff for an Asian put option
        payoffs.append(max(K - avg_price, 0))
    
    # Calculate the average payoff and discount it back to the present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def monte_carlo_european_put_antithetic(S0, K, T, r, sigma, n_simulations=10000, steps=100):
    dt = T / steps
    payoffs = []

    for _ in range(n_simulations // 2):  # Halve the number of loops for paired antithetic paths
        S_t1, S_t2 = S0, S0  # Two paths: one regular, one antithetic
        for _ in range(steps):
            Z = np.random.normal()
            S_t1 = S_t1 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            S_t2 = S_t2 * np.exp((r - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt) * Z)
        
        # Calculate the payoff for a European put option at maturity
        payoff1 = max(K - S_t1, 0)
        payoff2 = max(K - S_t2, 0)
        
        # Average the payoff from the paired paths
        payoffs.append(0.5 * (payoff1 + payoff2))
    
    # Calculate the average payoff and discount it back to the present
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def implied_volatility_call(S, K, T, r, market_price, tol=1e-5, max_iterations=100):
    """Calculate the implied volatility of a European call option using the Newton-Raphson method."""
    sigma = 0.2  # Initial guess for volatility
    min_sigma = 1e-4  # Minimum threshold to avoid division by zero
    
    for i in range(max_iterations):
        # Calculate price with current sigma
        price = black_scholes_call(S, K, T, r, sigma)
        
        # Calculate Vega (the derivative of the option price with respect to volatility)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        # Update sigma using Newton-Raphson method
        price_diff = price - market_price
        if abs(price_diff) < tol:
            return sigma  # Converged to a solution
        sigma -= price_diff / vega  # Update rule for Newton-Raphson
        
        # Ensure sigma does not fall below the minimum threshold
        sigma = max(sigma, min_sigma)
    
    # If no convergence, return NaN to indicate failure
    return np.nan

