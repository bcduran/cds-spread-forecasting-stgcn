# CDS Spread Forecasting with STGCN

This project implements a **Spatio-Temporal Graph Convolutional Network (STGCN)** to forecast CDS spread dynamics using supply chain network structure.

The pipeline combines:
- panel-based CDS time series
- graph-based dependency modeling through adjacency matrices
- benchmark models such as **Naive** and **AR(1)**
- out-of-sample evaluation
- protection-based long-short backtesting

The project is designed as a research-oriented machine learning pipeline for financial risk forecasting.

---

## Project Objective

The main goal is to predict CDS spread behavior using both:

1. **Temporal information** from historical CDS series
2. **Cross-sectional network structure** derived from supply chain relationships

The project compares a graph-based deep learning model against simpler statistical baselines and evaluates both predictive performance and trading/backtesting implications.

---

## Methodology Overview

The workflow includes:

- loading CDS panel data from `ve1.csv`
- loading the supply chain adjacency matrix from `adj.npz`
- normalizing the graph structure
- creating rolling historical input windows
- splitting the data into train / validation / test sets
- scaling the panel data using `StandardScaler`
- training:
  - Naive baseline
  - AR(1) baseline
  - STGCN model
- evaluating out-of-sample performance using:
  - MSE
  - RMSE
  - R²
- running a protection-based long-short backtest using predicted CDS changes

---

## Model Architecture

The STGCN model consists of:

- a **graph convolution layer** using normalized adjacency
- a **temporal convolution layer**
- a **fully connected output layer**

This allows the model to jointly capture:

- inter-firm relationships through the graph
- temporal dependence through rolling input windows

---

## Baseline Models

The following benchmark models are included:

### Naive
- In `LEVEL` mode: predicts the next level as the last observed level
- In `DELTA` mode: predicts no change

### AR(1)
- Fits a separate AR(1)-style relation for each firm
- Uses the last observed value in the rolling window for one-step-ahead prediction

### STGCN
- Learns graph-aware and temporal patterns jointly
- Supports both `LEVEL` and `DELTA` forecasting modes

---

## Backtesting

The project also evaluates prediction usefulness through a CDS protection-based backtest.

Portfolio construction logic:
- long protection on firms with the highest predicted widening
- short protection on firms with the strongest predicted tightening

Reported backtest statistics include:
- mean PnL
- volatility
- hit ratio
- Sharpe ratio

---

## File Structure

```text
cds-spread-forecasting-stgcn/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── README.md
├── outputs/
│   └── README.md
└── src/
    └── cds_stgcn_pipeline.py
