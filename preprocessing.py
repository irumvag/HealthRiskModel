from pipeline import process_newborn_data  

processed_df = process_newborn_data(
    r"datasets\newborn_health_monitoring_100k_rows.csv",
    r"datasets\Rwanda_Vaccine_Schedule.csv",
)
#print(processed_df.head())
#print(processed_df.columns)
