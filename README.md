# Product/Marketing Analysis & Churn and LTV Prediction
This repository contains my materials for a pet-project with product/marketing analysis. 
The project covers ETL pipelines using Airflow, thorough exploratory data analysis, RFM segmentation,
and prediction of churn and LTV using Cox's Proportional Hazard, BG/NBD, and Gamma-Gamma models.

The project uses a [**dataset from Kaggle**](https://www.kaggle.com/datasets/rishikumarrajvansh/marketing-insights-for-e-commerce-company/data).

# Table of Contents

**1. Database creation**
  The database was designed by me in accordance with 3NF.

  The ER diagram:

  ![er_diagram](https://github.com/maxim-lipatnikov/marketing-data-etl/blob/main/images/er_diagram.png)
  
**2. ETL**
  - Developing ETL pipeline in an Airflow DAG.

    The pipeline consists of:
      - Extracting the data straight from Kaggle using API
      - Transforming the data so that it fully corresponds with the database tables
      - Loading the transformed data into the database
  
**3. Downloading data from the database**

**4. Handling NaNs and changing data types**

**5. RFM segmentation**
  - Elbow method
  - K-Means

**6. Exploratory data analysis**
  - Tenure check
  - Customers by gender and location, MAU
  - Customer acquisition by month, CAC
  - Cohort analysis - retention and total revenue
  - Revenue and profit analysis, ARPU
  - Number of orders, AOV
  - Products, categories, and discounts analysis
  - Summary

**7. Churn prediction - survival analysis**
  - Using Cox's Proportional Hazard Model to predict tenure

**8. LTV prediction - BG/NBD and Gamma-Gamma**
  - Comparing Gamma-Gamma method to tenure method

# Tech Stack
- Python with libraries (airflow, pandas, numpy, matplotlib, seaborn, lifetimes, lifelines, sklearn)
- Jupyter Notebook

# Results
Upon completion of this project, I gained a deeper understanding of product and marketing analytics. 
The hands-on experience enhanced my skills in data analysis and applying different methods (RFM, K-Means, CoxPH, BG/NBD, Gamma-Gamma models) for customer segmentation
and prediction. This notebook, as well as .py files, serves as a full outline of the project and a reference for similar future projects.
