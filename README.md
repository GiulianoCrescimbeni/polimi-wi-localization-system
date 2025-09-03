# WI - Indoor Localization System Using RSSI

This project implements and evaluates a Wi-Fi-based indoor localization system using RSSI (Received Signal Strength Indicator) data. The system is developed for the *Wireless Internet* (WI) course at Politecnico di Milano.

## Overview

The goal is to localize a device indoors by analyzing Wi-Fi signal strengths received from access points. The system was implemented and tested in a real residential environment, where signal data was collected, cleaned, and processed for training and evaluation of various machine learning algorithms.

## Features

- Preprocessing pipeline including:
  - Data cleaning
  - Dummy entry insertion
  - Data augmentation
- Heatmap visualization of MAC address signal distributions
- Implementation of 4 ML models:
  - **Probabilistic**: Horus, Bayesian Network
  - **Deterministic**: K-Nearest Neighbors (KNN), Random Forest
- Evaluation using:
  - Confusion matrices
  - Classification accuracy
  - Mean localization error (in meters)
 
## Visualization

### House Floor Plan

This grid was used during the data acquisition phase to collect RSSI values at predefined positions.

<p align="center">
  <img src="HousePlant/HousePlantSamplingPoints.png" alt="House Plant" width="750"/>
</p>

### Heatmap Example

This heatmap shows the signal intensity distribution for one of the selected MAC addresses.

<p align="center">
  <img src="Images/HeatMap.png" alt="Heat Map" width="600"/>
</p>

### KNN Confusion Matrix

The confusion matrix illustrates the excellent performance of the KNN classifier.

<p align="center">
  <img src="Images/KNNMatrix.png" alt="KNN Matrix" width="600"/>
</p>

## Results Summary

| Algorithm         | Accuracy | Mean Localization Error |
|------------------|----------|--------------------------|
| Horus            | 0.5377   | 1.07 m                   |
| Bayesian Network | 0.7722   | 0.46 m                   |
| KNN              | 0.9766   | 0.09 m                   |
| Random Forest    | 0.9672   | 0.08 m                   |

## Project Structure

- **HousePlant/**  
  Includes floor plan files of the environment used for data collection, with annotated sampling points.

- **SampleData/**  
  Original raw data files (PCAP and unprocessed CSV), organized by measurement position.

- **ProcessedData/**  
  Contains datasets that have been cleaned, filled with dummy values, and augmented as needed—these are used as input for model training.

- **BalancedData/**  
  Contains datasets balanced across sampling positions to ensure uniform training conditions.

- **LocalizationAlgorithms/**  
  Includes Python scripts implementing the four machine learning algorithms (Horus, Bayesian Network, KNN, Random Forest), with training and evaluation logic.

- **ProcessingScripts/**  
  Utility Python scripts used during the offline phase:
  - PCAP to CSV conversion  
  - Signal cleaning and normalization  
  - Dummy entry insertion  
  - Data augmentation tools  

- **WN_IndoorLocalizationSystem.pdf**  
  Final report of the project, with detailed explanation of the methodology, implementation, and results.

---

## Documentation

For full implementation details, methodology, and analysis, refer to the [project report](./WN_IndoorLocalizationSystem.pdf).
