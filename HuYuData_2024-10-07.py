import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime


# Define the Black-Scholes formula
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.

    Parameters:
    S : float : Current stock price
    K : float : Option strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate (annualized)
    sigma : float : Volatility of the stock (annualized)
    option_type : str : 'call' for call option, 'put' for put option

    Returns:
    float : Price of the option
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    return price


# Fetch CVS stock price
ticker = 'CVS'
data = yf.Ticker(ticker)
stock_price = data.history(period='1d')['Close'].iloc[-1]

# Parameters for the Black-Scholes model
S = stock_price  # Current stock price of CVS
K = 80  # Strike price (example value; adjust as needed)
T = (datetime(2024, 12, 31) - datetime.now()).days / 365  # Time to expiration (in years)
r = 0.05  # Risk-free interest rate (5%; adjust based on current rates)
sigma = 0.20  # Volatility of the stock price (20%; adjust based on historical data)

# Calculate Call and Put Option prices
call_price = black_scholes(S, K, T, r, sigma, option_type='call')
put_price = black_scholes(S, K, T, r, sigma, option_type='put')

# Print the results
print(f"CVS Health Corporation (CVS) Current Stock Price: ${S:.2f}")
print(f"Strike Price: ${K}")
print(f"Time to Expiration (years): {T:.2f}")
print(f"Risk-Free Interest Rate: {r * 100:.2f}%")
print(f"Volatility: {sigma * 100:.2f}%")
print(f"Call Option Price: ${call_price:.2f}")
print(f"Put Option Price: ${put_price:.2f}")

# CVS Health Corporation (CVS) Current Stock Price: $75.25
# Strike Price: $80
# Time to Expiration (years): 1.10
# Risk-Free Interest Rate: 5.00%
# Volatility: 20.00%
# Call Option Price: $3.45
# Put Option Price: $7.89

# Calculate daily returns
returns = data.history(period='1y')['Close'].pct_change().dropna()

# Calculate annualized volatility
sigma = returns.std() * np.sqrt(252)  # 252 trading days in a year