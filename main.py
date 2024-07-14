import argparse
import pandas as pd
from src.prob_measures import p1, p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str)
    args = parser.parse_args()
    
    df = pd.read_csv(args.dataset)
    
    print("P1.X:", round(p1(df, "x"), 6))
    print("P1.X:", round(p1(df, "y"), 6))
    print("P2:", round(p2(df), 6))
