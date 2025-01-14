# train.py
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def train(input_file='sim_output.pkl', output_file='train_output.pkl'):
    with open(input_file, 'rb') as f:
        (X_labeled, y_labeled) = pickle.load(f)
   
    X_input = X_labeled.reshape(-1, 1) 
    model = MLPRegressor(
        hidden_layer_sizes=(32, 32, 16, 16),  
        activation='relu',                     
        solver='adam',                         
        max_iter=1000,                         
        learning_rate='adaptive',              
        random_state=42                        
    )
    model.fit(X_input, y_labeled)
    
    y_pred = model.predict(X_input)
    mse = mean_squared_error(y_labeled, y_pred)
    
    print(f"Training completed. MSE: {mse:.4f}")

    with open(output_file, 'wb') as f:
        pickle.dump(((X_labeled, y_labeled), model), f)
    
    print(f"Model saved to {output_file}")
    return output_file, mse

if __name__ == "__main__":
    train()  # Running the training task
