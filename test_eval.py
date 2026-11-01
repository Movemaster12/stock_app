from model import load_stock_data, prepare_data, PredictionModel, train_model, evaluate_model, device
import numpy as np

df = load_stock_data('^GSPC', '2020-01-01')
X_train, y_train, X_test, y_test, scaler = prepare_data(df, seq_length=30)

model = PredictionModel(input_dim=1, hidden_dim=64, num_layers=4, output_dim=1).to(device)
train_model(model, X_train, y_train, num_epochs=200, learning_rate=0.05)
results = evaluate_model(model, X_train, y_train, X_test, y_test, scaler)

mape = np.mean(np.abs((results['test_actual'][:,0] - results['test_predictions'][:,0]) / results['test_actual'][:,0])) * 100
naive_pred = results['test_actual'][:-1, 0]
naive_actual = results['test_actual'][1:, 0]
naive_mape = np.mean(np.abs((naive_actual - naive_pred) / naive_actual)) * 100
naive_mape_zero = np.mean(np.abs((results['test_actual'][:,0] - 0) / results['test_actual'][:,0])) * 100
naive_rmse_zero = np.sqrt(np.mean(results['test_actual'][:,0]**2))
model_rmse = results['test_rmse']

# print(f"Model MAPE: {mape:.3f}%")
# print(f"Naive baseline MAPE: {naive_mape:.3f}%")
print(f"Model RMSE (returns): {model_rmse:.6f}")
print(f"Naive (predict-zero) RMSE (returns): {naive_rmse_zero:.6f}")
print("Prediction std:", np.std(results['test_predictions'][:,0]))
print("Actual std:", np.std(results['test_actual'][:,0]))