import pandas as pd

# Step 1: Load the CSV file
message_file_path = '/Users/szorathgera/Dissertation/AlphaTrade/data/messageFiles/AMZN_2012-06-21_34200000_57600000_message_10.csv'
orderbook_file_path = '/Users/szorathgera/Dissertation/AlphaTrade/data/orderbookFiles/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'

message_data = pd.read_csv(message_file_path, header=None)
orderbook_data = pd.read_csv(orderbook_file_path, header=None)

# Step 2: Calculate the index for the 70-30 split
message_split_index = int(len(message_data) * 0.7)
orderbook_split_index = int(len(orderbook_data)* 0.7)


# Step 3: Split the data into training and testing datasets
message_train_data = message_data[:message_split_index]  # First 70% for training
print(message_train_data)
message_test_data = message_data[message_split_index:]   # Last 30% for testing
print(message_test_data)
orderbook_train_data = orderbook_data[:orderbook_split_index]
orderbook_test_data = orderbook_data[orderbook_split_index:]

# Step 4: Save the split data into separate CSV files
message_train_file_path = '/Users/szorathgera/Dissertation/AlphaTrade/data_train/messageFiles/AMZ_message_train_10.csv'
message_test_file_path = '/Users/szorathgera/Dissertation/AlphaTrade/data_test/messageFiles/AMZ_message_test_10.csv'

orderbook_train_file_path = '/Users/szorathgera/Dissertation/AlphaTrade/data_train/orderbookFiles/AMZ_orderbook_train_10.csv'
orderbook_test_file_path = '/Users/szorathgera/Dissertation/AlphaTrade/data_test/orderbookFiles/AMZ_orderbook_test_10.csv'

message_train_data.to_csv(message_train_file_path, index=False, header=False)
message_test_data.to_csv(message_test_file_path, index=False, header=False)

orderbook_train_data.to_csv(orderbook_train_file_path, index=False, header=False)
orderbook_test_data.to_csv(orderbook_test_file_path, index=False, header=False)

print(f"Message Training data saved to {message_train_file_path}")
print(f"Message Testing data saved to {message_test_file_path}")

print(f"Orderbook Training data saved to {orderbook_train_file_path}")
print(f"Orderbook Testing data saved to {orderbook_test_file_path}")
