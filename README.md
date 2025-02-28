# NTT Data Hackathon - Energy Markets Analysis

This repository contains our team's solutions for the NTT Data Hackathon focused on energy market analysis. The hackathon involved tackling two key challenges in the energy sector using data science and machine learning approaches.

## Project Structure

```
.
├── challenge_one/ (To be added later)
│   └── ...
├── challenge_two/
│   ├── Challenge2_404TeamNotFound.ipynb
│   ├── README.md
│   └── dashboard.py
├── EDA/
│   └── eda+fe_Aswin.ipynb
└── doc/
    ├── NTT DATA Hackathon IE vpresented.pdf
    └── merging_df_ideas.txt
```

## Challenge Overview

### Challenge One: Energy Consumption Forecasting (To be added later)

This challenge focused on predicting energy consumption patterns. The solution will be added at a later date.

### Challenge Two: Market Code Prediction

In this challenge, we developed a sequence-to-sequence (Seq2Seq) model to automatically assign market codes to anonymized energy bid data. Our approach leverages LSTM networks to understand temporal patterns in energy consumption and generate the corresponding market code sequences.

Key features of our solution:
- **Preprocessing Pipeline**: Converted temporal data and engineered features to capture cyclical patterns
- **Encoder-Decoder Architecture**: Used LSTM networks to process variable-length sequences
- **Teacher Forcing**: Implemented to stabilize training and accelerate convergence
- **High Accuracy**: Achieved approximately 89% accuracy on unseen data

For more details, see the [Challenge Two README](challenge_two/README.md).

## Team Members

- 404 Team Not Found

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Streamlit (for dashboard visualization)

## Running the Code

### Prerequisites

Install the required dependencies:

```bash
pip install pandas numpy streamlit plotly torch scikit-learn
```

### Running the Jupyter Notebooks

```bash
jupyter notebook *.ipynb
```

### Running the Dashboard

```bash
streamlit run challenge_two/dashboard.py
```

## Acknowledgments

Special thanks to NTT Data for organizing this hackathon and providing the datasets and problem statements that allowed us to explore energy market dynamics.