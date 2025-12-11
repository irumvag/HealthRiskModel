import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 100_000

# Realistic baby names (Rwanda + common)
first_names = [
    'Alice','Jean','Marie','Pierre','Aline','Eric','Grace','David','Sarah','John',
    'Yvette','Patrick','Claude','Olivier','Diane','Emma','Noah','Liam','Olivia','Sophia'
]

surnames = [
    'Niyonsenga','Mukamana','Habimana','Tuyizere','Uwera','Habyarimana','Iradukunda',
    'Ishimwe','Nkurunziza','Kwizera','Mutesi','Uwineza','Gakwaya','Nsabimana','Keza'
]

# age limits
n_years_max = 5
ref_date = datetime(2024, 12, 1)

# -------------------------------------------------------------------
# Helper: approximate WHO-like targets for weight (kg) and length (cm)
#  - birth  ~3.3 kg, 50 cm
#  - 1 year ~9.5-10 kg, 75 cm
#  - 5 years ~18-19 kg, 110 cm
# -------------------------------------------------------------------
def target_growth(age_years, gender):
    # 0 years
    if age_years <= 0:
        wt0_m, ht0_m = 3.3, 50.0
        wt0_f, ht0_f = 3.2, 49.5
        return (wt0_m, ht0_m) if gender == 'Male' else (wt0_f, ht0_f)

    # 0–1 year: linear between birth and 1y
    if age_years <= 1:
        if gender == 'Male':
            wt0, ht0 = 3.3, 50.0
            wt1, ht1 = 9.8, 75.0
        else:
            wt0, ht0 = 3.2, 49.5
            wt1, ht1 = 9.2, 73.0
        t = age_years / 1.0
        return (wt0 + t * (wt1 - wt0), ht0 + t * (ht1 - ht0))

    # 1–5 years: linear between 1y and 5y
    if gender == 'Male':
        wt1, ht1 = 9.8, 75.0
        wt5, ht5 = 18.5, 110.0
    else:
        wt1, ht1 = 9.2, 73.0
        wt5, ht5 = 18.0, 109.0

    t = min(max((age_years - 1.0) / 4.0, 0.0), 1.0)
    return (wt1 + t * (wt5 - wt1), ht1 + t * (ht5 - ht1))

data = []

# -------------------------------------------------------------------
# Generate synthetic data
# -------------------------------------------------------------------
for babyid in range(1, n + 1):
    gender = np.random.choice(['Male', 'Female'])

    # birth between 2018-01-01 and 5 years before reference date
    start_birth = datetime(2018, 1, 1)
    birth_date = start_birth + timedelta(days=np.random.randint(0, 365 * n_years_max))

    # age in days at measurement date with small jitter
    agedays_exact = (ref_date - birth_date).days + np.random.randint(-30, 30)
    agedays = max(0, min(agedays_exact, int(n_years_max * 365)))

    age_years = agedays / 365.25

    # approximate WHO-like targets
    target_wt, target_len = target_growth(age_years, gender)

    # head circumference: ~34-35 cm at birth, ~47-51 cm at 5y
    if gender == 'Male':
        hc_birth, hc_5y = 35.0, 50.0
    else:
        hc_birth, hc_5y = 34.5, 49.0
    hc = hc_birth + (hc_5y - hc_birth) * min(age_years / 5.0, 1.0)

    # add natural variation
    weightkg = round(max(1.5, np.random.normal(target_wt, 0.8)), 3)
    lengthcm = round(max(38.0, np.random.normal(target_len, 3.0)), 2)
    headcircumferencecm = round(max(30.0, np.random.normal(hc, 1.5)), 2)

    # birth values (reasonable ranges)
    birthweight_rough = np.random.normal(3.2 if gender == 'Female' else 3.3, 0.5)
    birthweightkg = round(float(np.clip(birthweight_rough, 2.0, 4.5)), 3)

    birthlength_rough = np.random.normal(49.5 if gender == 'Female' else 50.0, 2.0)
    birthlengthcm = round(float(np.clip(birthlength_rough, 45.0, 55.0)), 2)

    data.append({
        'babyid': babyid,
        'name': np.random.choice(first_names) + ' ' + np.random.choice(surnames),
        'gender': gender,
        'gestationalageweeks': np.random.choice([36, 37, 38, 39, 40, 41, 42]),
        'birthweightkg': birthweightkg,
        'birthlengthcm': birthlengthcm,
        'birthheadcircumferencecm': round(np.random.normal(34.0, 1.5), 2),
        'date': (birth_date + timedelta(days=agedays)).strftime('%Y-%m-%d'),
        'agedays': agedays,
        'weightkg': weightkg,
        'lengthcm': lengthcm,
        'headcircumferencecm': headcircumferencecm,
        'temperature': round(np.random.normal(36.7, 0.4), 2),
        'heartratebpm': int(np.clip(np.random.normal(130, 15), 90, 190)),
        'respiratoryratebpm': int(np.clip(np.random.normal(35, 7), 20, 70)),
        'oxygensaturation': int(np.clip(np.random.normal(98, 1), 90, 100)),
        'feedingtype': np.random.choice(['Breastfed', 'Formula', 'Mixed']),
        'feedingfrequencyperday': np.random.randint(4, 12),
        'urineoutputcount': np.random.randint(4, 12),
        'stoolcount': np.random.randint(1, 10),
        'jaundicelevelmgdl': round(max(0.0, np.random.normal(4.0, 3.0)), 2),
        'apgarscore': np.random.choice([7, 8, 9, 10]),
        'immunizationsdone': np.random.choice(['Yes', 'Partial', 'No'], p=[0.8, 0.15, 0.05]),
        'reflexesnormal': np.random.choice(['Yes', 'No'], p=[0.94, 0.06]),
        'risklevel': np.random.choice(['Low', 'Medium', 'High'], p=[0.75, 0.20, 0.05])
    })

df = pd.DataFrame(data)

# -------------------------------------------------------------------
# WHO-based validation (0–24 months only)
#   - You must have the WHO CSVs in the same folder:
#       WHO-Boys-Weight-for-age-Percentiles.csv
#       WHO-Girls-Length-for-age-Percentiles.csv
# -------------------------------------------------------------------
who_boys_wfa = pd.read_csv("WHO-Boys-Weight-for-age-Percentiles.csv")   # boys weight-for-age 0–24m [file:22]
who_girls_lfa = pd.read_csv("WHO-Girls-Length-for-age-Percentiles.csv") # girls length-for-age 0–24m [file:21]

who_boys_wfa = who_boys_wfa.rename(columns={"Month": "MonthNum"})
who_girls_lfa = who_girls_lfa.rename(columns={"Month": "MonthNum"})

# build lookup dicts by month
boys_weight_lookup = {
    int(row.MonthNum): {
        "p2": row["2nd (2.3rd)"],
        "p98": row["98th (97.7th)"],
        "median": row["50th"],
    }
    for _, row in who_boys_wfa.iterrows()
}

girls_length_lookup = {
    int(row.MonthNum): {
        "p2": row["2nd (2.3rd)"],
        "p98": row["98th (97.7th)"],
        "median": row["50th"],
    }
    for _, row in who_girls_lfa.iterrows()
}

def clamp_month(m):
    return max(0, min(int(m), 24))

# limit to 0–24 months (~0–730 days)
mask_0_24m = df["agedays"] <= 730
df_0_24m = df[mask_0_24m].copy()
df_0_24m["agemonths"] = df_0_24m["agedays"] / 30.4375

# boys: weight-for-age validation in 0–24 months
boys_0_24 = df_0_24m[df_0_24m["gender"] == "Male"].copy()
boys_0_24["month_clamped"] = boys_0_24["agemonths"].astype(int).clip(0, 24)

boys_0_24["who_p2"] = boys_0_24["month_clamped"].apply(lambda m: boys_weight_lookup[m]["p2"])
boys_0_24["who_p98"] = boys_0_24["month_clamped"].apply(lambda m: boys_weight_lookup[m]["p98"])
boys_0_24["who_median"] = boys_0_24["month_clamped"].apply(lambda m: boys_weight_lookup[m]["median"])

boys_0_24["out_of_range"] = (
    (boys_0_24["weightkg"] < boys_0_24["who_p2"]) |
    (boys_0_24["weightkg"] > boys_0_24["who_p98"])
)

print("WHO validation 0-24 months (boys, weight-for-age):")
print("Total boys rows       :", len(boys_0_24))
print("Outside 2nd–98th pct  :", int(boys_0_24["out_of_range"].sum()))
print("Mean synthetic weight :", round(boys_0_24["weightkg"].mean(), 2))
print("Mean WHO median wt    :", round(boys_0_24["who_median"].mean(), 2))

# girls: length-for-age validation in 0–24 months
girls_0_24 = df_0_24m[df_0_24m["gender"] == "Female"].copy()
girls_0_24["month_clamped"] = girls_0_24["agemonths"].astype(int).clip(0, 24)

girls_0_24["who_p2"] = girls_0_24["month_clamped"].apply(lambda m: girls_length_lookup[m]["p2"])
girls_0_24["who_p98"] = girls_0_24["month_clamped"].apply(lambda m: girls_length_lookup[m]["p98"])
girls_0_24["who_median"] = girls_0_24["month_clamped"].apply(lambda m: girls_length_lookup[m]["median"])

girls_0_24["out_of_range"] = (
    (girls_0_24["lengthcm"] < girls_0_24["who_p2"]) |
    (girls_0_24["lengthcm"] > girls_0_24["who_p98"])
)

print("\nWHO validation 0-24 months (girls, length-for-age):")
print("Total girls rows      :", len(girls_0_24))
print("Outside 2nd–98th pct  :", int(girls_0_24["out_of_range"].sum()))
print("Mean synthetic length :", round(girls_0_24["lengthcm"].mean(), 2))
print("Mean WHO median len   :", round(girls_0_24["who_median"].mean(), 2))

# -------------------------------------------------------------------
# Save dataset
# -------------------------------------------------------------------
df.to_csv('newborn_health_monitoring_100k_rows.csv', index=False)
print("\nDataset created successfully!")
print(f"Rows   : {len(df):,}")
print("File   : newborn_health_monitoring_100k_rows.csv")
print("Ready to use with your full pipeline")
