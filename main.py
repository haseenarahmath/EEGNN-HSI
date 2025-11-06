#!/usr/bin/env python3
from src.core.parser import build_parser
from src.core.runner import run

def main():
    args = build_parser("EEGNN — Early-Exit GNN for HSI (ISDA 2023)").parse_args()
    run(args)

if __name__ == "__main__":
    main()
