# Sales Prediction Using Machine Learning

This project predicts sales from advertising spend on TV, Radio, and Newspaper using Machine Learning.
Compared multiple regression models including Linear Regression, Ridge, Lasso, Random Forest, and Gradient
Boosting using R2 Score, MAE, RMSE, and cross-validation.
Built a Streamlit web app to allow interactive sales prediction from user-entered advertising budgets.

## Files

- `sales_prediction_improved.ipynb` - cleaned and improved notebook workflow
- `app.py` - Streamlit app for interactive sales prediction
- `requirements.txt` - Python dependencies


The notebook also supports the original Colab path:

```text
/content/drive/MyDrive/project/input.csv
```

## Improvements Included

- Clean notebook structure
- Data understanding and EDA
- Simple TV-only baseline model
- Multiple model comparison
- R2, MAE, and RMSE evaluation
- 5-fold cross-validation
- Feature importance
- Actual vs predicted plot
- Residual analysis
- Final prediction function
- Streamlit app

## Run Notebook

Open `sales_prediction_improved.ipynb` and run the cells from top to bottom.

## Run App

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py --server.address 127.0.0.1
```
