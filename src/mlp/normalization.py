import numpy as np
from sklearn.preprocessing import MinMaxScaler

# X_train: Training gene expression matrix
# X_other: Validation or test expression matrix
def normalize_data(X_train, X_other): # 
    scaler = MinMaxScaler()

    # Learn minimum and maximum values from training
    X_train_scaled = scaler.fit_transform(X_train)

    X_other_scaled = scaler.transform(X_other)

    return (
        X_train_scaled.astype(np.float32),
        X_other_scaled.astype(np.float32),
        scaler
    )
