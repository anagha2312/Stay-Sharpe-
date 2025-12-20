import pandas as pd
import numpy as np
import sys

def create_mock_dataframe(rows=100):
    timestamps = np.arange(1664582400, 1664582400 + rows * 60, 60) # 1-minute intervals
    data = {
        'timestamp': timestamps,
        'open': np.random.uniform(98, 102, size=rows),
        'high': np.random.uniform(102, 105, size=rows),
        'low': np.random.uniform(95, 98, size=rows),
        'close': np.random.uniform(98, 102, size=rows),
        'volume': np.random.uniform(1000, 5000, size=rows)
    }
    df = pd.DataFrame(data)
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    return df

def print_failure(message, details=""):
    """Prints a standardized failure message and exits."""
    print(f"\n[TEST FAILED] {message}")
    if details:
        print(f"    Details: {details}")
    print("\nScript execution halted. Please fix the issue above to proceed.")
    sys.exit(1)

def run_tests():
    print("[*] Step 1: Importing 'CtrlAlpha' from 'ctrl_alpha.py'...")
    try:
        sys.path.append('.')
        from ctrl_alpha import CtrlAlpha
        print("[SUCCESS] Step 1: Class imported successfully.\n")
    except ImportError:
        print_failure("Could not import the script.", "Make sure your file is named 'ctrl_alpha.py' and is in the same directory.")
    except AttributeError:
        print_failure("Could not find class 'CtrlAlpha' in your script.", "Please check that the class is named exactly 'CtrlAlpha'.")
    except Exception as e:
        print_failure("An unexpected error occurred during import.", f"{e}")

    print("[*] Step 2: Instantiating the CtrlAlpha class...")
    try:
        model = CtrlAlpha()
        print("[SUCCESS] Step 2: Class instantiated successfully.\n")
    except Exception as e:
        print_failure("Failed to instantiate the 'CtrlAlpha' class.", f"Check your class __init__() method. Error: {e}")

    print("[*] Step 3: Calling the train() function...")
    try:
        mock_df = create_mock_dataframe()
        model.train(train_df=mock_df)
        print("[SUCCESS] Step 3: train() function executed without errors.\n")
    except Exception as e:
        print_failure("An error occurred while calling the train() function.", f"Error: {e}")

    print("[*] Step 4: Calling the predict() function...")
    try:
        mock_row = mock_df.iloc[-1]
        mock_timestamp = int(mock_row['timestamp'])
        signal = model.predict(row=mock_row, timestamp=mock_timestamp)
        print(f"    -> predict() returned: {signal}")
        print("[SUCCESS] Step 4: predict() function executed without errors.\n")
    except Exception as e:
        print_failure("An error occurred while calling the predict() function.", f"Error: {e}")
    
    print("[*] Step 5: Validating the output of the predict() function...")
    if not isinstance(signal, int) or signal not in [-1, 0, 1]:
        print_failure(
            "The return value from predict() is in the wrong format.",
            f"Expected an integer (-1, 0, or 1), but got type {type(signal)} with value {signal}."
        )
    print("[SUCCESS] Step 5: predict() function returned a valid signal.\n")

    print("All I/O tests passed successfully!")
    print("Disclaimer: This script only checks for formatting and basic I/O. It does not evaluate the quality of your prediction logic.")

run_tests()