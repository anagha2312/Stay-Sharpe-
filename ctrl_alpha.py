import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier 
import logging
import sys
import warnings 


logging.basicConfig(level=logging.INFO, stream=sys.stdout)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')


EXECUTION_DELAY = 10 
PROFIT_MULTIPLIER = 1.5
LOSS_MULTIPLIER = 1.0
FEATURE_WINDOW = 20
CONFIDENCE_THRESHOLD = 0.55


class CtrlAlpha:
    """
    Final optimized script, using robust parameters for balance between 
    prediction quality and low execution latency.
    """

    def __init__(self):
        """Initialize all persistent objects and configuration settings."""
    
        self.scaler = None
        self.model = None

     
        self.execution_delay = EXECUTION_DELAY
        self.feature_window = FEATURE_WINDOW
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.profit_multiplier = PROFIT_MULTIPLIER
        self.loss_multiplier = LOSS_MULTIPLIER
        
      
        self.max_history = self.feature_window * 3 
        self.close_history = np.array([])
        
      
        self.label_map = {-1: 0, 0: 1, 1: 2}
        
  
        self.feature_cols = ['SMA', 'VOLATILITY', 'Momentum', 'Dist_SMA']
        
        logging.info("CtrlAlpha initialized.")

 

    def _engineer_features_train(self, df):
        """Generates all features using optimized Pandas methods (fast on large data)."""
        df = df.copy()

    
        df['Returns'] = df['close'].pct_change()
    

        df['SMA'] = df['close'].rolling(window=self.feature_window, min_periods=self.feature_window).mean()
        
 
        df['VOLATILITY'] = df['Returns'].rolling(window=self.feature_window, min_periods=self.feature_window).std()
        

        df['Momentum'] = df['close'].diff(self.feature_window)
        
 
        df['Dist_SMA'] = df['close'] - df['SMA']
        
        features = df[self.feature_cols].copy()
        
 
        return features.fillna(0)


    def _get_triple_barrier_labels(self, df):
        """Generates the target labels (1, -1, 0) for training."""
        close = df['close'].values
        N = len(close)
        labels = np.zeros(N, dtype=int)
        
   
        vol_df = df['close'].pct_change().rolling(window=20, min_periods=10).std().fillna(0)

        for t in range(N - self.execution_delay):
            entry_price = close[t]
            volatility_t = vol_df.iloc[t]
            
            upper_barrier = entry_price * (1 + self.profit_multiplier * volatility_t)
            lower_barrier = entry_price * (1 - self.loss_multiplier * volatility_t)
            
            for k in range(1, self.execution_delay + 1):
                i = t + k
                if i >= N: break 
                
                current_price = close[i]
                
                if current_price >= upper_barrier:
                    labels[t] = 1 
                    break
                elif current_price <= lower_barrier:
                    labels[t] = -1
                    break
            
        mapped_labels = np.vectorize(self.label_map.get)(labels)
        target = pd.Series(mapped_labels, index=df.index, dtype=int)
        
        return target.iloc[:-self.execution_delay] 



    def train(self, train_df: pd.DataFrame):
        """Model training."""
        logging.info("Starting probabilistic model training...")
        
        df_copy = train_df.copy()
        
        for col in ['r_h', 'y_h']:
            if col in df_copy.columns:
                df_copy = df_copy.drop(columns=[col])
        
  
        X_full = self._engineer_features_train(df_copy)
        

        y_full = self._get_triple_barrier_labels(df_copy) 

   
        X = X_full.loc[y_full.index]
        y = y_full.astype(int)

    
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
     
        self.model = XGBClassifier(
            objective='multi:softmax', 
            num_class=3,
            eval_metric='mlogloss',
            n_estimators=50,
            max_depth=2, 
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=1 
        )
        
        self.model.fit(X_scaled, y)
        logging.info("Training Complete.")


    def predict(self, row: pd.Series, timestamp: int) -> int:
        """Generates the signal. EXTREMELY OPTIMIZED for ultra-low latency."""
        
        if self.model is None or self.scaler is None:
            return 0 

    
        if 'close' not in row.index:
            return 0
            
        new_close_price = row['close'] 
        

        self.close_history = np.append(self.close_history, new_close_price)
        if len(self.close_history) > self.max_history:
            self.close_history = self.close_history[-self.max_history:]
        
        N = len(self.close_history)
        
   
        if N < self.feature_window + 1:
            return 0 
        
   
        W = self.feature_window
        
        prices_for_features = self.close_history[-W:]
        prices_for_returns = self.close_history[-W-1:]
        
 
        SMA = prices_for_features.mean()
        
     
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.diff(prices_for_returns) / prices_for_returns[:-1]
            VOLATILITY = np.nan_to_num(np.std(returns)) 
        
      
        Momentum = self.close_history[-1] - self.close_history[-W-1]
        
     
        Dist_SMA = self.close_history[-1] - SMA
        
     
        X_single = np.array([[SMA, VOLATILITY, Momentum, Dist_SMA]])
        
       
        X_scaled = self.scaler.transform(X_single)

      
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probabilities = self.model.predict_proba(X_scaled)[0]
        
        prob_buy = probabilities[self.label_map[1]]   
        prob_sell = probabilities[self.label_map[-1]] 
        
   
        if prob_buy > self.confidence_threshold:
            signal = 1
        elif prob_sell > self.confidence_threshold:
            signal = -1
        else:
            signal = 0 
            
        return int(signal)
