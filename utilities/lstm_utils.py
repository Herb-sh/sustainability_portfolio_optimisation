import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' 4.3. LSTM '''
'''
Both train and test sequences are identical=12, but the lasts (x=1,3,12) months in the test
data are ahead of training data
'''
def split_train_test(df_time_series, df_static=None, in_seq_length=12, out_seq_length=12, validation_months=5*12):
    # Set sequence length (e.g., 12 time points)

    # Create sequences for the normalized data
    if df_static is None:
        df_static = []
    X, X_static, y = create_sequences(df_time_series, df_static, in_seq_length, out_seq_length)

    # skip first item
    '''
    X_train = X[:len(X) - (validation_months - out_seq_length)]
    X_test = X[(len(X) - (validation_months - out_seq_length)):]

    y_train = y[:(len(X) - (validation_months - out_seq_length))]
    y_test = y[(len(X) - (validation_months - out_seq_length)):]
    '''
    X_train = X[:len(X) - (validation_months)]
    X_test = X[(len(X) - (validation_months)):]

    y_train = y[:(len(X) - (validation_months))]
    y_test = y[(len(X) - (validation_months)):]

    return X_train, X_test, y_train, y_test

"""
    Create sequences of data for LSTM model.
"""
def create_sequences(df_ts, df_static, seq_length, out_seq_length=1):
    x_ts, x_static, y = [], [], []
    # starting from 0, input & output sequences should be excluded from total dataframe length
    for i in range(len(df_ts) - seq_length - out_seq_length): # -1
        x_ts_data = df_ts.iloc[i: i+seq_length].values
        x_ts_data_transposed = x_ts_data.transpose(1, 0)

        #
        if len(df_static) > 0:
            x_static.append(df_static[i]) # Sequences are NOT added to
        #
        y_data = df_ts.iloc[i+seq_length: i + seq_length + out_seq_length].values
        y_data_transposed = y_data.transpose(1, 0)

        # y_data = fix_array_length(y_data, out_seq_length)
        if len(y_data) == out_seq_length:
            x_ts.append(x_ts_data_transposed)  # Sequence of `seq_length` time points
            y.append(y_data_transposed)   # Target is the next time step

    return (torch.tensor(x_ts, dtype=torch.float32),
            torch.tensor(x_static, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


def lstm_train_validate(model, optimizer, X_train, X_test, y_train, y_test):
    EPOCHS = 10 # 5 # 100
    batch_size = 32

    loss_fn = nn.MSELoss()

    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=False, batch_size=batch_size)
    test_loader = data.DataLoader(data.TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_size)

    y_train_pred = torch.tensor([])
    y_test_pred = torch.tensor([])

    y_train_pred_batch = torch.tensor([])
    y_test_pred_batch = torch.tensor([])

    # Training & Test Loop
    for epoch in range(EPOCHS):
        # reset all predictions on each epoch
        y_train_pred = torch.tensor([])
        y_test_pred = torch.tensor([])

        model.train()
        total_loss = 0

        # Train
        for index, (X_batch, y_batch) in enumerate(train_loader):
            y_train_pred_batch = torch.tensor([]) # reset predictions
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if torch.any(torch.isnan(X_batch)):
                print("NaN values found during training: X_batch: ", epoch, index, X_batch[0][0])
                continue
            if torch.any(torch.isnan(y_batch)):
                print("NaN values found during training: y_batch: ", epoch, index, y_batch[0][0])
                continue
            # Forward pass
            optimizer.zero_grad()

            y_train_pred_batch = model(X_batch)

            # Compute loss
            loss = loss_fn(y_train_pred_batch, y_batch)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            # Join all batch predictions together, only last epoch where model is fully trained
            y_train_pred = torch.cat([y_train_pred, y_train_pred_batch], dim=0)

        # Validation - Root-mean-square-error
        model.eval()

        # Test
        with torch.no_grad():
            for index, (X_batch, y_batch) in enumerate(test_loader):

                y_test_pred_batch = torch.tensor([]) # reset predictions

                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if torch.any(torch.isnan(X_batch)):
                    print("NaN values found during validation. X_batch: ", epoch, index, X_batch[0][0])
                    continue
                if torch.any(torch.isnan(y_batch)):
                    print("NaN values found during validation. y_batch: ", epoch, index, y_batch[0][0])
                    continue

                y_test_pred_batch = model(X_batch).squeeze(-1)
                # Join all batch predictions together
                y_test_pred = torch.cat([y_test_pred, y_test_pred_batch], dim=0)
            #

            train_rmse = np.sqrt(loss_fn(y_train_pred, y_train))
            test_rmse = np.sqrt(loss_fn(y_test_pred, y_test))
        # Print epoch loss
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}. ")

    # Save the trained model
    torch.save(model.state_dict(), "lstm_univariate.pth")
    print("Model training complete and saved.")

    return model, y_train_pred, y_test_pred

'''
Convert Prophet forecast to a dataframe
'''
def get_df_from_forecast(forecast):
    # Collect 'ds' (date) and 'yhat' from each forecast
    forecast_dfs = [item[['ds', 'yhat']].rename(columns={'yhat': stock}) for stock, item in forecast.items()]

    # Merge all forecasts on 'ds' (date)
    merged_forecast = forecast_dfs[0]
    for df in forecast_dfs[1:]:
        merged_forecast = merged_forecast.merge(df, on='ds', how='outer')

    merged_forecast = merged_forecast.set_index('ds')

    return merged_forecast