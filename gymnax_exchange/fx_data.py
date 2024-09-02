#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 14:48:04 2024

@author: Michael Birkenzeller

    This code intends to reproduce the files provided by LOBSTER, while
    enabling a higher flexibility in terms of the created message data, 
    as well as the depth of the created limit order book.
    
    Work in progress! If you find errors or improve any parts of the
    code, I´m happy to receive those updates.
    
    (https://lobsterdata.com)

	Message File:		(Matrix of size: (Nx6))
	-------------	
			
	Name: 	Messages.csv 	
    
	Columns:
	
	    1.) Time: 		
				So far just a running integer number, where the time delta
                between trades is based on a Poisson Process.
                
	    2.) Type:
				1: Submission of a new limit order
				2: Cancellation (Partial deletion of a limit order)
                    Only partial cancellation possible, if quantity of
                    cancelled shares surpasses book --> rest quantity of 1
				3: Deletion (Total deletion of a limit order)
                    NOT implemented (keep probability to 0)
				4: Execution of a visible limit order			   	 
				5: Execution of a hidden limit order
                    NOT implemented
				7: Trading halt indicator 	
                    NOT implemented 
                    
	    3.) Order ID: 	
				Random Number - possibility of repetition
                
	    4.) Size: 		
				Number of shares
                
	    5.) Price: 		
				Dollar price times 10000 
				(i.e., A stock price of $91.14 is given 
				by 911400)
                
	    6.) Direction:
				-1: Sell limit order
				1: Buy limit order
				
				Note: 
				Execution of a sell (buy) limit
				order corresponds to a buyer (seller) 
				initiated trade, i.e. Buy (Sell) trade.
										
						
	Orderbook File:		(Matrix of size: (Nx(4xNumberOfLevels)))
	---------------
	
	Name: 	OrderBook.csv
	
	Columns:
	
 	    1.) Ask Price 1: 	Level 1 Ask Price 	(Best Ask)
	    2.) Ask Size 1: 	Level 1 Ask Volume 	(Best Ask Volume)
	    3.) Bid Price 1: 	Level 1 Bid Price 	(Best Bid)
	    4.) Bid Size 1: 	Level 1 Bid Volume 	(Best Bid Volume)
	    5.) Ask Price 2: 	Level 2 Ask Price 	(2nd Best Ask)
	    ...
    
"""

# Import Libraries
import numpy as np
import pandas as pd

#%% Definition formulas

# definition Variable Lamda
def update_lambda(lambda_current, volatility):
    
    dt = 1
    lambda_mean=0.2
    theta=0.5
    sigma=1.5
    
    # Adjust lambda_mean or sigma based on volatility
    adjusted_lambda_mean = lambda_mean * (1 + volatility)  # Example adjustment
    adjusted_sigma = sigma * (1 + volatility)  # Example adjustment

    # Ornstein-Uhlenbeck process for lambda (mean-reverting)
    dW = np.random.normal(0, np.sqrt(dt))
    lambda_new = lambda_current + theta * (adjusted_lambda_mean - lambda_current) * dt + adjusted_sigma * dW

    return max(lambda_new, 0.05)

class RollingVolatility:
    def __init__(self, window=100):
        self.window = window
        self.values = []
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0

    def update(self, new_price):
        # Calculate the return from the previous price
        if self.count > 0:
            last_price = self.values[-1]
            if last_price != 0:
                new_return = (new_price - last_price) / last_price
            else:
                new_return = 0  # Safeguard against division by zero
        else:
            new_return = 0  # Initial value (or use a small non-zero if preferred)

        # If the window is full, remove the oldest value
        if self.count >= self.window:
            old_return = self.values.pop(0)
            self.count -= 1
            delta = old_return - self.mean
            self.mean -= delta / self.count
            self.M2 -= delta * (old_return - self.mean)
        
        # Update the rolling statistics with the new return
        self.values.append(new_return)
        self.count += 1
        delta = new_return - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (new_return - self.mean)

    @property
    def volatility(self):
        if self.count < 2:
            return 0.0
        variance = self.M2 / (self.count - 1)
        return np.sqrt(variance)
    
# definition Order Generation
def generate_order(TypeDict, startTime, tickSize,
                   current_bid, current_ask,
                   lambda_current=0.01, rollingVolatility=0.5):

    currentMidPrice = (current_bid + current_ask) // 2 
    
    lambda_rate = update_lambda(lambda_current, rollingVolatility) #0.1
    inter_arrival_time = np.random.exponential(1 / lambda_rate)
    
    order_time = startTime + inter_arrival_time
    
    order_type = np.random.choice(list(TypeDict.keys()), p=list(TypeDict.values()))

    direction = 1 if np.random.rand() >= 0.5 else -1

    order_size = np.random.randint(5, 50)

    if order_type != 4:  # For Limit Orders, Cancellations, and Deletions        
        dist_mid = abs(int(np.random.normal(0, tickSize * 10)))
        price = currentMidPrice - (dist_mid * direction)
        price = round(price / tickSize) * tickSize

    else:  # For Executions
        if direction == -1:  # Sell execution
            price = current_bid
        else:  # Buy execution
            price = current_ask
    
    # Create the order message
    order_message = {
        'Time': order_time,
        'Type': order_type,
        'OrderID': np.random.randint(1e6, 1e7),
        'Size': order_size,
        'Price': price,
        'Direction': direction if order_type in [1, 4] else 0
    }
    
    return order_message, lambda_rate

#%% Creation message and LOB data

params = {
    "LOBsize":100000,
    "InitBid":1100000,
    "InitAsk":1108300,
    "TickSize":100,
    "StandingVol":4000,
    "TypeDict":{1: 0.5, 2: 0.2, 3: 0.0, 4: 0.3},
    "StartTime": 0,
    "Levels": 1,
    }

# init LOB
levels_per_Side = 10
volAtLev = params["StandingVol"] / (levels_per_Side*2)
tick = params["TickSize"]
LOB = np.zeros((params['LOBsize'], levels_per_Side*4))
for i in range(0, levels_per_Side*4, 4):
    LOB[0][i] = params["InitAsk"] + tick*(i/4)
    LOB[0][i+1] = volAtLev
    LOB[0][i+2] = params["InitBid"] - tick*(i/4)
    LOB[0][i+3] = volAtLev

messages = np.zeros((params['LOBsize'], 6))
midPrice = np.zeros((params['LOBsize'], 1))
midPrice[0] = (params["InitAsk"] + params["InitBid"]) // 2
rolling_vol = RollingVolatility(window=100)
lambdaR = 0.01
lambdaCheck = []

for i in range(1, len(LOB)-1):
    
    order, lambdaR = generate_order(
        params["TypeDict"], params["StartTime"], params["TickSize"],
        LOB[i-1][2], LOB[i-1][0],
        lambdaR, rolling_vol.volatility
        )
    
    messages[i-1] = [
        order['Time'], order['Type'], order['OrderID'],
        order['Size'], order['Price'], order['Direction']
        ]

    def implementBuy(index, i, price, quant): #BID
        newLine = LOB[i-1].copy()
        p, q = 4*levels_per_Side-2, 4*levels_per_Side-1
        while p > index and p > 4:
            newLine[p] = (LOB[i-1][p-4]).copy()
            newLine[q] = (LOB[i-1][q-4]).copy()
            p -= 4
            q -= 4
        newLine[p], newLine[q] = price, quant
        return newLine
    
    def implementSell(index, i, price, quant): #ASK
        newLine = LOB[i-1].copy()
        p, q = 4*levels_per_Side-4, 4*levels_per_Side-3
        while p > index and p > 2:
            newLine[p] = (LOB[i-1][p-4]).copy()
            newLine[q] = (LOB[i-1][q-4]).copy()
            p -= 4
            q -= 4
        newLine[p], newLine[q] = price, quant
        return newLine
    
    # handling BUY Limit Orders
    if order['Direction']==1 and order['Type']==1: #BID
        level = 1
        while level <= levels_per_Side:
            step = level*4-2
            if order['Price'] > LOB[i-1][step]:
                newLine = implementBuy(step, i, order['Price'], order['Size'])
                LOB[i] = newLine
                break
            elif order['Price'] == LOB[i-1][step]:
                newLine = LOB[i-1].copy()
                newLine[step+1] += order['Size']
                LOB[i] = newLine
                break
            level += 1
        else:
            LOB[i] = LOB[i-1].copy()
    
    # handling SELL Limit Orders
    elif order['Direction']==-1 and order['Type']==1:
        level = 1
        while level <= levels_per_Side:
            step = level*4-4
            if order ['Price'] < LOB[i-1][step]:
                newLine = implementSell(step, i, order['Price'], order['Size'])
                LOB[i] = newLine
                break
            elif order['Price'] == LOB[i-1][step]:
                newLine = LOB[i-1].copy()
                newLine[step+1] += order['Size']#
                LOB[i] = newLine
                break
            level += 1
        else:
            LOB[i] = LOB[i-1].copy()
                
    elif order['Type'] == 2:
        newLine = LOB[i-1].copy()
        if order['Direction'] == 1:
            if order['Price'] in newLine[2::4]:
                index = newLine[2::4].tolist().index(order['Price'])
                index = 2 + index * 4
                if order['Size'] < newLine[index + 1]:
                    newLine[index + 1] -= order['Size']
                else:
                    newLine[index + 1] = 1     
        elif order['Direction'] == -1:
            if order['Price'] in newLine[0::4]:
                index = newLine[0::4].tolist().index(order['Price'])
                index = index * 4
                if order['Size'] < newLine[index + 1]:
                    newLine[index + 1] -= order['Size']
                else:
                    newLine[index + 1] = 1 
        LOB[i] = newLine
        
    elif order['Direction']==1 and order['Type']==4:
        if order['Size'] < LOB[i-1][1]:
            newLine = LOB[i-1].copy()
            newLine[1] -= order['Size']
        else:
            newLine = LOB[i-1].copy()
            for j in range(0, 4*levels_per_Side-5, 4):
                newLine[j] = newLine[j + 4]
                newLine[j + 1] = newLine[j + 5]
            newLine[4*levels_per_Side-4] = newLine[12]+tick
            newLine[4*levels_per_Side-3] = 5
        LOB[i] = newLine
            
    elif order['Direction']==-1 and order['Type']==4:
        if order['Size'] < LOB[i-1][3]:
            newLine = LOB[i-1].copy()
            newLine[3] -= order['Size']
        else:
            newLine = LOB[i-1].copy()
            for j in range(2, 4*levels_per_Side-3, 4):
                newLine[j] = newLine[j + 4]
                newLine[j + 1] = newLine[j + 5]
            newLine[4*levels_per_Side-2] = newLine[14]-tick
            newLine[4*levels_per_Side-1] = 5
        LOB[i] = newLine
        
    params["StartTime"] = order['Time']
    midPrice[i] = (LOB[i][0] + LOB[i][2]) // 2

    rolling_vol.update(midPrice[i-1])
    lambdaCheck.append(lambdaR)

# Adjust data to get rid of first 100 steps computed with a fix lambda
messageData = messages[100:-2]
LOBData = LOB[100:-2]
midPriceProcess = midPrice[100:-2]
    
#%% Save the data
path_message = "/Users/szorathgera/Dissertation/AlphaTrade/data_train/fx_message/"
path_orderbook = "/Users/szorathgera/Dissertation/AlphaTrade/data_train/fx_orderbook/"



np.set_printoptions(suppress=True)
np.savetxt(path_message+'MessageData.csv', messageData, delimiter=',', fmt='%.18f')
np.savetxt(path_orderbook+'LOBData.csv', LOBData, delimiter=',', fmt='%d')


#%% Adjust Time
'''
General easy Idea:
    - Input: start and end date 
    - Input: trading times on each day (e.g. 9am to 5pm or 24h/7...)
    
    - Take lowest and higest value and divide to fit the desired days
    - The same then for the daily times...
    
Cooler Version:
    - do not simply create data and then adjust times but take the time in and
    use it to adjust trading behavior during the day, e.g.:
        - wider spreads in the beginning of the day (higher standard dev. of orders)
        - more volume in the beginning and especially EOD (adjust lambda process)


#%% Simple Implementation

import numpy as np
import pandas as pd
from datetime import datetime, timedelta



# Example start date
start_date = datetime(2024, 1, 1)   # Starting on Jan 1, 2024
nDays = 3                        # Number of Trading Days

# Trading hours (9 AM to 4 PM)
trading_start = timedelta(hours=9)   # 9 AM
trading_end = timedelta(hours=16)    # 4 PM

# Calculate the number of trading days
trading_days = pd.bdate_range(start=start_date, periods=nDays).to_pydatetime()
total_trading_days = len(trading_days)

# Calculate total time spent in the trading session each day in seconds
trading_seconds_per_day = (trading_end - trading_start).total_seconds()

# Scale the time column in messages to fit within the trading day time range
total_simulation_time = messageData[-1, 0]  # Assuming the last value in the time column is the total time
time_per_day = total_simulation_time / total_trading_days

# Scale and adjust the time values
adjusted_times = []
for i, time_value in enumerate(messageData[:, 0]):
    # Determine which trading day this time value corresponds to
    day_index = int(time_value // time_per_day)
    intra_day_time = (time_value % time_per_day) / time_per_day  # Normalize to [0, 1]
    
    if day_index >= total_trading_days:
        day_index = total_trading_days - 1  # Clamp to the last day
    
    # Map to a specific trading time within the day
    adjusted_seconds = intra_day_time * trading_seconds_per_day
    actual_time = trading_days[day_index] + trading_start + timedelta(seconds=adjusted_seconds)
    
    # Convert actual_time to seconds since epoch
    adjusted_times.append(actual_time.timestamp())

# Replace the original time column with adjusted timestamp values
adjusted_times = np.array(adjusted_times)
messageData = messageData.astype(float)  # Use float type for storing timestamps
messageData[:, 0] = adjusted_times

# Now messageData[:, 0] contains adjusted timestamps that fit within the specified trading hours.

#%% Print Chart

midPriceProcess[-1]

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Convert the timestamps back to datetime objects for plotting
x = [datetime.fromtimestamp(ts) for ts in messageData[:, 0]]
y = midPriceProcess

# Plot the lines
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Mid Price Process", linewidth=1, color="C0")

# Add some Titles and notations
plt.title("Price Chart")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

# Format the x-axis to show dates sparsely
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

# Rotate the x-axis labels for better readability
plt.gcf().autofmt_xdate()

# Display Everything
plt.show()
    

#%% # OLD PRINT



import matplotlib.pyplot as plt
import matplotlib.dates as mdates

x = messageData[:, 0]
y = midPriceProcess

# Plot the lines
plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Mid Price Process", linewidth=1, color="C0")

# Add some Titles and notations
plt.title("Price Chart")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

# Format the x-axis to show dates sparsely
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

# Optionally, rotate the x-axis labels for better readability
plt.gcf().autofmt_xdate()

# Display Everything
plt.show()

#%% OLD Adjust TIME

messageData = messages[100:-2]
LOBData = LOB[100:-2]
midPriceProcess = midPrice[100:-2]

# Easy Version:

from datetime import datetime, timedelta

# Example start and end dates
start_date = datetime(2024, 1, 1)  # Starting on Jan 1, 2024
end_date = datetime(2024, 1, 1)   # Ending on Jan 5, 2024

# Trading hours (9 AM to 4 PM)
trading_start = timedelta(hours=9)   # 9 AM
trading_end = timedelta(hours=16)    # 4 PM

# Calculate the number of trading days
trading_days = pd.bdate_range(start=start_date, end=end_date).to_pydatetime()
total_trading_days = len(trading_days)

# Calculate total time spent in the trading session each day in seconds
trading_seconds_per_day = (trading_end - trading_start).total_seconds()

# Scale the time column in messages to fit within the trading day time range
total_simulation_time = messageData[-1, 0]  # Assuming the last value in the time column is the total time
time_per_day = total_simulation_time / total_trading_days

# Scale and adjust the time values
adjusted_times = []
for i, time_value in enumerate(messageData[:, 0]):
    # Determine which trading day this time value corresponds to
    day_index = int(time_value // time_per_day)
    intra_day_time = (time_value % time_per_day) / time_per_day  # Normalize to [0, 1]
    
    if day_index >= total_trading_days:
        day_index = total_trading_days - 1  # Clamp to the last day
    
    # Map to a specific trading time within the day
    adjusted_seconds = intra_day_time * trading_seconds_per_day
    actual_time = trading_days[day_index] + trading_start + timedelta(seconds=adjusted_seconds)
    
    adjusted_times.append(actual_time)

# Replace the original time column with adjusted datetime values
adjusted_times = np.array(adjusted_times)
messageData = messageData.astype(object)
messageData[:, 0] = adjusted_times

# Convert the datetime array back to string or timestamp for easier viewing (optional)
messageData[:, 0] = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in adjusted_times]

#%%

'''
























