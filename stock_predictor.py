import streamlit as st
from stock_utils import fetch_stock_info, stock_name
from model import PredictionModel, load_stock_data, prepare_data, train_model, evaluate_model, device
import pandas as pd
import matplotlib.pyplot as plt

st.title('Stocks Predictor')

with st.sidebar:
    st.header("Model Configuration")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    st.subheader("Model Parameters")
    seq_length = st.slider(
        "Sequence Length (days)", min_value=10,value=30,max_value=60, help="Number of days to look back")
    hidden_dim = st.slider("Hidden Dimensions", min_value=16,value=32,max_value=128, help="Size of LSTM hidden state")
    num_layer=st.slider("LSTM Layers", min_value=1,value=2,max_value=4, help="Number of stacked layers")
    num_epochs=st.slider("Training Epochs", min_value=50,value=200, max_value=500, step=50)
    learning_rate = st.select_slider("Learning Rate", options=[0.01, 0.05, 0.075], value=0.01)
    train_button = st.button(
        "🚀 Train & Predict", 
        type="primary", 
        use_container_width=True
    )


my_symbol = st.text_input('Enter a stock ticker', '^GSPC', key='predictor') # S&P 500 is default

if my_symbol and my_symbol.strip():
    try:
        information = fetch_stock_info(my_symbol)
        stock_name(information,my_symbol)

        st.divider()        

          
        # train_button = st.button("Train & Preduct", type="primary", use_container_width=True)
        if train_button:
            st.subheader("Loading Data")
            with st.spinner(f"Downloading data for {my_symbol}"):
                try:
                    df = load_stock_data(my_symbol, start_date)
                    st.success(f"Loaded {len(df)} days of data")
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Days", len(df))
                with col2:
                    st.metric("Latest Price", f"{df['Close'].iloc[-1]:.2f}")
                with col3:
                    st.metric("Highest", f"{df['Close'].max():.2f}")
                with col4:
                    st.metric("Lowest", f"{df['Close'].min():.2f}")

                st.subheader("Preparing Data")
                with st.spinner("Creating sequences"):
                    try:
                        X_train, y_train, X_test, y_test,scaler = prepare_data(df, seq_length)
                        st.success("Data prepared successfully")
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.stop()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Samples", len(X_train))
                with col2:
                    st.metric("Test Samples", len(X_test))
                with col3:
                    train_pct = len(X_train) / (len(X_train) + len(X_test)) * 100
                    st.metric("Split", f"{train_pct:.0f}% / {100-train_pct:.0f}%")

                st.subheader("Initialising Model")
                model = PredictionModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layer, output_dim=1).to(device)

                total_params = sum(p.numel() for p in model.parameters())
                st.success(f"Model initialised with {total_params:,} parameters")

                st.subheader("Training Model")

                progress_bar = st.progress(0)
                status_text = st.empty()
                loss_chart_placeholder = st.empty()

                losses = []
                def training_callback(epoch, loss):
                    losses.append(loss)
                    progress = (epoch + 1) / num_epochs
                    progress_bar.progress(progress)
                    
                    if epoch % 25 == 0 or epoch == num_epochs - 1:
                        status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.6f}")
                        
                        if len(losses) > 1:
                            fig_loss, ax_loss = plt.subplots(figsize=(10, 3))
                            ax_loss.plot(losses, color='#d62728', linewidth=2)
                            ax_loss.set_xlabel('Epoch')
                            ax_loss.set_ylabel('Loss (MSE)')
                            ax_loss.set_title('Training Loss')
                            ax_loss.grid(True, alpha=0.3)
                            loss_chart_placeholder.pyplot(fig_loss)
                            plt.close(fig_loss)
                
            with st.spinner("Training in progress..."):
                train_losses = train_model(
                    model, X_train, y_train,
                    num_epochs, learning_rate,
                    callback=training_callback
                )
            
            progress_bar.progress(1.0)
            st.success("Training compelete!")

            st.subheader("Model Evaluation")
            with st.spinner("Evaluating..."):
                results = evaluate_model(
                    model, X_train, y_train, X_test, y_test, scaler
                )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training RMSE", f"${results['train_rmse']:.2f}")
            with col2:
                st.metric("Test RMSE", f"${results['test_rmse']:.2f}")

            st.subheader("Prediction vs Actual")
            fig_pred, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Training predictions
            ax1.plot(results['train_actual'][:, 0], 
                    label='Actual', color='#2ca02c', linewidth=2)
            ax1.plot(results['train_predictions'][:, 0], 
                    label='Predicted', color='#ff7f0e', 
                    linewidth=2, alpha=0.7, linestyle='--')
            ax1.set_title('Training Set', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Test predictions
            ax2.plot(results['test_actual'][:, 0], 
                    label='Actual', color='#2ca02c', linewidth=2)
            ax2.plot(results['test_predictions'][:, 0], 
                    label='Predicted', color='#ff7f0e', 
                    linewidth=2, alpha=0.7, linestyle='--')
            ax2.set_title('Test Set', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Price ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_pred)

            st.subheader("Recent Predictions")
            n_show = min(10, len(results['test_actual']))
            pred_df = pd.DataFrame({
                'Actual Price': results['test_actual'][-n_show:, 0],
                'Predicted Price': results['test_predictions'][-n_show:, 0],
            })
            pred_df['Difference'] = pred_df['Actual Price'] - pred_df['Predicted Price']
            pred_df['Difference %'] = (pred_df['Difference'] / pred_df['Actual Price'] * 100)
            
            st.dataframe(
                pred_df.style.format({
                    'Actual Price': '${:.2f}',
                    'Predicted Price': '${:.2f}',
                    'Difference': '${:.2f}',
                    'Difference %': '{:.2f}%'
                }),
                use_container_width=True)
            
            st.info("Configure parameters in the sidebar and click **'Train & Predict'**")

    except Exception as e:
        st.error(f'Error: {e}')
else:
    st.info('Please enter a stock ticker symbol to get started')