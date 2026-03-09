# °‧ 𓆝 𓆟 𓆞 ·｡ ElectroFish Imitation Model °‧ 𓆝 𓆟 𓆞 ·｡

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Machine Learning](https://img.shields.io/badge/Continual_Learning-Active-success)

## Overview

The **ElectroFish** project focuses on studying a unique species of fish equipped with an electric organ. These fish emit electric impulses primarily for communication and navigation. A general goal of this research is to gain deeper insights into the complex social interactions of these fish.

This software project specifically aims to examine and model the behavioral patterns of the fish—encompassing both their physical movement trajectories and their electric signaling—using **Continual Learning** methodologies. 

## Key Concepts

For human observers, fish behavior can be understood through visible patterns such as varying speeds, swimming angles, or preferred locations within the tank. We capture these behaviors using two main types of features:

* **Low-Level Features:** Distance to the partner fish, distance to the tank walls, relative angles, and instantaneous speed.
* **High-Level Features:** Temporal behaviors observed over multiple timestamps, such as *‘following a partner fish’* or *‘circling in a corner’*.

## Project Structure

The repository is structured to handle the entire machine learning pipeline, from data ingestion to trajectory prediction:

* `prepare_data.py`: Scripts for cleaning, formatting, and feature-engineering the raw data (extracting low/high-level features).
* `analyse_data.py`: Tools for exploratory data analysis (EDA) and visualizing the fish's movement and electric signaling patterns.
* `train_model.py`: The main training loop utilizing Continual Learning techniques to model behavioral patterns.
* `predict_trajectory.py`: Inference script used to predict future movements or behaviors based on the trained model.
* `LSTM_original/`: Directory containing the initial/baseline Long Short-Term Memory (LSTM) network implementation.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed.

```bash
# Clone the repository
git clone https://github.com/maybe-im-a-mess/ElectroFish_ImitationModel.git
cd ElectroFish_ImitationModel
```

### Usage

1. **Prepare the Data:** Download the dataset (see below) and run the preparation script to extract relevant features.
   ```bash
   python prepare_data.py
   ```
2. **Analyze:** Get visual insights into the tank environment and fish interactions.
   ```bash
   python analyse_data.py
   ```
3. **Train the Model:** Train the imitation model using the extracted features.
   ```bash
   python train_model.py
   ```
4. **Predict Trajectories:** Test the model's accuracy on unseen data.
   ```bash
   python predict_trajectory.py
   ```

## Data Source

The dataset used for training and evaluating these models contains tracking and signaling data for the fish. 

**[Download the Dataset Here](https://drive.google.com/drive/folders/11kz1BGG4qCm7uGdHDINltBo53fD_YvPD?usp=drive_link)** *(Google Drive Link)*

---
*Created for the analysis of electric fish behavior using Continual Learning.*
