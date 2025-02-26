# Post-Mortem Report: Challenge Two – Market Code Prediction

This report outlines the approach, design decisions, and key results for Challenge Two. Our objective was to automatically assign market codes to anonymized energy bid data (df_omie_blind) by leveraging historical labeled data (df_omie_labelled) using a sequence-to-sequence (Seq2Seq) model. Below, we detail our methodology, technical implementation, and the business impact of our solution.



## 1. Overview

**Objective:**
Develop a machine learning model that predicts market codes from energy consumption sequences. The goal was to associate each anonymized bid with a market code, thereby automating the analysis process for energy market bidding.

**Business Impact:**
- **Efficiency:** Automates the code assignment process, reducing manual intervention and expediting bid analysis.
- **Accuracy:** Achieves robust performance (≈89% accuracy on unseen data), ensuring high confidence in the automated predictions.
- **Scalability:** Designed to handle variable-length time sequences, making it adaptable to fluctuating market conditions and diverse datasets.



## 2. Data Preparation and Preprocessing

### Data Sources
- **df_omie_labelled.csv:** Contains historical bid data with market codes.
- **df_omie_blind.csv:** Provides anonymized bid data without market codes.
- **Supplementary Files:** Additional context (filtered categories and unit lists) was available to enrich our analysis.

### Key Preprocessing Steps
1. **Parsing and Cleaning:**
   - Converted datetime fields and extracted temporal features (e.g., hour, day of week, month).
   - Removed records where `price` or `energy` was zero to maintain data integrity.

2. **Feature Engineering:**
   - Applied sine/cosine transformations on cyclical features (hour, day of week) to capture periodic patterns.
   - Calculated percentiles for energy and price within each hour to standardize variability.

3. **Chunking the Dataset:**
   To facilitate sequence learning, the data was segmented into hourly chunks. This ensured that each sequence captured the energy behavior within a consistent time interval.

   *Example Code Snippet:*
   ```python
   def chunk_dataset(df):
       chunks = []
       unique_hours = df['hour'].unique()
       unique_months = df['month'].unique()
       unique_days = df['day_of_month'].unique()
       for month in unique_months:
           for day in unique_days:
               for hour in unique_hours:
                   chunk = df[(df['hour'] == hour) & (df['month'] == month) & (df['day_of_month'] == day)]
                   if not chunk.empty:
                       chunks.append(chunk)
       return chunks
   ```
   *Explanation:* This function organizes the dataset into temporally coherent chunks, allowing the model to learn energy patterns on an hourly basis.



## 3. Model Architecture and Rationale

### Overall Approach
We adopted an **encoder-decoder (Seq2Seq) architecture** using LSTM networks:
- **Encoder:** Processes sequences of energy values to extract meaningful temporal features.
- **Decoder:** Generates corresponding sequences of market code tokens.

### Encoder Design
- **Purpose:** Capture the dynamics of energy consumption.
- **Mechanism:** Utilizes an LSTM layer, enhanced by packing padded sequences to efficiently handle variable-length inputs.

   *Example Code Snippet:*
   ```python
   class Encoder(nn.Module):
       def __init__(self, input_dim, hidden_dim, num_layers=1):
           super(Encoder, self).__init__()
           self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

       def forward(self, x, lengths):
           packed = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
           _, (hidden, cell) = self.lstm(packed)
           return hidden, cell
   ```
   *Explanation:* Packing the sequences prevents the LSTM from processing padded tokens, improving training efficiency.

### Decoder Design
- **Purpose:** Convert encoded features into market code sequences.
- **Mechanism:** Incorporates an embedding layer for token representation, an LSTM layer for sequence generation, and a fully connected layer to output token probabilities.

   *Example Code Snippet:*
   ```python
   class Decoder(nn.Module):
       def __init__(self, output_dim, hidden_dim, num_layers=1):
           super(Decoder, self).__init__()
           self.embedding = nn.Embedding(output_dim, hidden_dim)
           self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
           self.fc = nn.Linear(hidden_dim, output_dim)

       def forward(self, input_token, hidden, cell):
           input_token = input_token.unsqueeze(1)
           embedded = self.embedding(input_token)
           output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
           prediction = self.fc(output.squeeze(1))
           return prediction, hidden, cell
   ```
   *Explanation:* This design enables the decoder to generate sequences token by token, with each prediction informed by the previous state.

### Teacher Forcing
During training, **teacher forcing** was used to provide the ground truth token as input to the decoder with a defined probability. This technique:
- Accelerates convergence.
- Stabilizes the training process.
- Helps mitigate the compounding of errors during sequence generation.



## 4. Training Process and Hyperparameter Tuning

### Data Splitting
- **Chronological Split:**
  - **Training (70%)** and **Validation (10%)** sets were created from earlier data to ensure no future information leaked during training.
  - **Test Set (20%)**: Held out for final evaluation.

### Training Details
- **Loss Function:** CrossEntropyLoss (ignoring padding tokens).
- **Optimizer:** Adam with a learning rate of 0.001.
- **Gradient Clipping:** Employed to prevent exploding gradients.
- **Epochs:** Initially trained for 10 epochs, later fine-tuned with 15 epochs using optimal hyperparameters.

   *Example Code Snippet:*
   ```python
   for epoch in range(NUM_EPOCHS):
       model_best.train()
       for src, trg, src_lengths in train_loader:
           src, trg = src.to(device), trg.to(device)
           optimizer_best.zero_grad()
           output = model_best(src, src_lengths, trg)
           loss = criterion_best(output[:, 1:].reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))
           loss.backward()
           torch.nn.utils.clip_grad_norm_(model_best.parameters(), max_norm=1)
           optimizer_best.step()
   ```
   *Explanation:* The loop iterates over batches, computes the loss on the predicted sequence (ignoring the start token), and updates model parameters accordingly.

### Hyperparameter Tuning
We experimented with:
- **Hidden Dimensions:** Evaluated 128 vs. 256 units; 256 yielded better performance.
- **Teacher Forcing Ratios:** Ranged from 0.5 to 0.9; a 0.9 ratio proved most effective.
- **Learning Rates:** Tested 0.001 and 0.0005, selecting 0.001 based on validation performance.



## 5. Inference on the Blind Dataset

After training, the final model was applied to the anonymized bid data. Each energy sequence from the blind dataset was processed to predict the corresponding market code sequence.

*Example Inference Function:*
```python
def predict_sequence(model, energy_seq, device, max_len=50, sos_token=SOS_TOKEN, eos_token=EOS_TOKEN):
    model.eval()
    src_tensor = torch.tensor(energy_seq, dtype=torch.float).unsqueeze(1).unsqueeze(0).to(device)
    src_length = torch.tensor([len(energy_seq)], dtype=torch.long).to(device)
    hidden, cell = model.encoder(src_tensor, src_length)
    input_token = torch.tensor([sos_token], dtype=torch.long).to(device)
    predicted_tokens = []
    for _ in range(max_len):
        output, hidden, cell = model.decoder(input_token, hidden, cell)
        predicted_token = output.argmax(1).item()
        if predicted_token == eos_token:
            break
        predicted_tokens.append(predicted_token)
        input_token = torch.tensor([predicted_token], dtype=torch.long).to(device)
    return predicted_tokens
```
*Explanation:* This function processes each energy sequence iteratively, stopping when an end-of-sequence token is generated or the maximum length is reached. The predictions are then mapped back to the original market codes using our label encoder.



## 6. Conclusion and Future Directions

### Key Achievements
- **Automated Code Assignment:** Successfully predicted market codes for anonymized energy bids with robust performance.
- **Model Robustness:** The LSTM-based Seq2Seq architecture, coupled with teacher forcing and effective preprocessing, ensured stable training and high prediction accuracy.
- **Business Value:** Streamlined the bid analysis process, enabling quicker and more reliable market decisions.

### Lessons Learned and Future Work
- **Feature Engineering:** Further exploration of additional temporal or market-related features may improve model performance.
- **Model Enhancements:** Experimenting with more advanced architectures (e.g., Transformer-based models) could capture even more nuanced patterns in the data.
- **Operational Integration:** Next steps include integrating the model into a real-time bidding system to support dynamic market decision-making.
