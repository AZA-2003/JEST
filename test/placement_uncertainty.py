import JEST
import argparse
from pathlib import Path



if __main__ == "__name__":
    parser = argparse.ArgumentParser(description='JEST Placement Uncertainty')
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
