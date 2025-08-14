

## Project Overview
This project simulates daily sales for **InnovateMart** stores and trains a **Temporal Fusion Transformer (TFT)** model to forecast the next 7 days of sales.

---

## Project Structure
``` bash
. ├── data/ 
│ └── simulated_innovatemart_daily_sales.csv # Simulated sales data 
│ └── *.png # Marcket dialy sales plot 
├── checkpoints/ 
│ └── BEST_CHECKPOINT.txt # Path to the latest trained model 
│ └── bes_model*.ckp # Best trained model 
├── generate_data.py # Generate data for sales prediction 
├── stream.py # Streamlit UI for sales prediction 
├── model.py # TFT model training script 
├── config.py # City and population configuration 
└── README.md # This file
```


---

## Screenshot
![UI Screenshot](data\UI.png)  
---

## Installation(Step1)
1. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
## Install required libraries:
```bash
pip install -r requirements.txt
```

Main libraries:

pandas, numpy, matplotlib, plotly
streamlit
torch, pytorch-lightning
pytorch-forecasting


## Simulate Sales Data(Step2)
``` bash
python generate_data.py --n_stores 5 --start_date 2022-01-01 --end_date 2024-12-31 --save ./data
```

* --n_stores : Number of stores
* --start_date & --end_date : Simulation date range
* --save : Path to save generated data


## Train TFT Model(Step3)
``` bash
python model.py --n_stores 5 --start_date 2022-01-01 --end_date 2024-12-31 --save ./checkpoints
```
* Model is trained for all stores together
* Best model checkpoint is saved in BEST_CHECKPOINT.txt

## Run Streamlit UI(Step4)
``` bash
streamlit run stream.py
```
Features:

* Select a store and see historical sales
* Forecast 7-day sales
* Compare predictions with actual data
* View variable importance (feature importance)

## Important Notes
* Model path is automatically read from BEST_CHECKPOINT.txt
* You can manually change the model path if needed
* GPU is recommended for faster predictions, but CPU works too
