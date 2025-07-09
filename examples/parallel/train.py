# train.py
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train(input_file='sim_output.pkl', output_file='train_output.pkl'):
    # Load labeled data
    with open(input_file, 'rb') as f:
        (X_labeled, y_labeled), _ = pickle.load(f)
    
    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(X_labeled, y_labeled)
    
    # Predict and compute Mean Squared Error (MSE) as a simple performance metric
    y_pred = model.predict(X_labeled)
    mse = mean_squared_error(y_labeled, y_pred)
    
    print(f"Training completed. MSE: {mse:.4f}")

    # Save the trained model
    with open(output_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {output_file}")
    return output_file, mse

if __name__ == "__main__":
    train()  # Running the training task
