import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)


class InnovateMart:
    """
    Represent a single InnovateMart store located in a city.

    Attributes
    city : str
    The city name where the store is located.
    size : str
    Store size category; expected values: "small", "medium", "large".
    population : int
    Population of the city (used as a factor in sales calculation).
    base_mean : float
    A heuristic estimate of the store's base daily sales. Calculated from
    store size and city population to provide a baseline for simulation.

    Methods
    promotion()
    Placeholder for promotional-effects logic (e.g., temporary sales uplift).
    seasonality()
    Placeholder for seasonal adjustments (e.g., weekly/annual patterns).
    competitor_store()
    Placeholder for competitor impact modeling (e.g., cannibalization).
    """
    def __init__(self, store_city, store_size, city_population, dates, competitor_open_date=None):
        self.city = store_city
        self.size = store_size
        self.population = city_population 
        self.dates = dates
        size_factor = {"small": 1.0, "medium": 1.6, "large": 3.0}[store_size]
        self.base_mean = 100 * size_factor * (1 + 0.1 * city_population)  # عددی فرضی برای فروش پایه روزانه
    
        self.annual_growth_rates = np.random.uniform(0.02, 0.08)
        self.competitor_open_date = competitor_open_date
        self.competitor_impact = np.random.uniform(0.12, 0.30)

 
        self.promo_dates = self.promotion()
        self.school_holidays, self.holiday_dates = self.holiday()

    def promotion(self):
        campaigns_per_year = np.random.randint(8, 16)
        all_possible = pd.date_range(start=start_date, end=end_date, freq="D")
        chosen = np.random.choice(all_possible, size=campaigns_per_year * ( (pd.to_datetime(end_date).year - pd.to_datetime(start_date).year) + 1 ), replace=False)
        promo_dates = []
        for d in chosen:
            length = np.random.randint(1, 5)
            rng = pd.date_range(start=d, periods=length, freq="D")
            promo_dates.extend(list(rng))
        promo_dates = sorted(list({pd.Timestamp(d).normalize() for d in promo_dates if (pd.Timestamp(start_date) <= pd.Timestamp(d) <= pd.Timestamp(end_date))}))
        return promo_dates

    def holiday(self):
        # تعریف چند تعطیلات کلیدی به‌صورت نمونه مانند عید و آخر سال  
        holiday_dates = set(pd.to_datetime([  
            "2022-01-01", "2022-01-02","2022-01-03","2022-01-04",
            "2022-01-12", "2022-01-13","2022-03-14","2022-03-15",
            "2022-11-22", "2022-12-29",
            "2023-01-01", "2023-01-02","2023-01-03","2023-01-04",
            "2023-01-12", "2023-01-13","2023-03-14","2023-03-15",
            "2023-11-22", "2023-12-29",
            "2024-01-01", "2024-01-02","2024-01-03","2024-01-04",
            "2024-01-12", "2024-01-13","2024-03-14","2024-03-15",
            "2024-11-22", "2024-12-29",
        ]))  
        # مدرسه: بازه‌های تعطیلی فرضی (برای سادگی چند بازه در سال)  
        school_holidays = set()  
        for year in range(pd.to_datetime(start_date).year, pd.to_datetime(end_date).year + 1):  
            summer_start = pd.to_datetime(f"{year}-04-01")  
            summer_end = pd.to_datetime(f"{year}-06-30")  
            rng = pd.date_range(summer_start, summer_end)  
            for d in rng:  
                school_holidays.add(d.normalize())  
            rng2 = pd.date_range(f"{year}-12-24", f"{year}-12-26")  
            for d in rng2:  
                school_holidays.add(d.normalize())
        return school_holidays, holiday_dates

    
    def seasonality(self, date):
        dow = date.weekday()  # Saturday=0,...,Friday=6
        if dow >= 5:  # Thursday(5), Friday(6)
            weekday_seasonality = 1.25  
        elif dow == 4:  # Friday
            weekday_seasonality = 1.10
        else:
            weekday_seasonality = 1.0
        
        month = date.month
        
        if month in [6, 7, 1, 12]:
            annual_seasonality = 1.30  
        elif month in [4, 5]:
            annual_seasonality = 0.95
        else:
            annual_seasonality = 1.00

        return weekday_seasonality, annual_seasonality
    

    def daily_income(self, date):
        weekday_seasonality, annual_seasonality, = self.seasonality(date)
        # Promotion
        promo_flag = 1 if date in self.promo_dates else 0
        if promo_flag:
            # increase %20-%70
            promo_lift = np.random.uniform(0.20, 0.70) * (1.0 if self.size == "small" else (0.9 if self.size=="medium" else 0.8))
        else:
            promo_lift = 0.0
        
        # holiday  
        holiday_flag = 1 if date.normalize() in self.holiday_dates else 0  
        if holiday_flag:  
            holiday_effect = 1.5       
        elif date.normalize() in self.school_holidays:  
            holiday_effect = 1.15    
        else:  
            holiday_effect = 1.0  

        competitor_flag = 1 if (self.competitor_open_date is not None and date >= self.competitor_open_date) else 0
        competitor_multiplier = (1 - self.competitor_impact) if competitor_flag else 1.0

        #
        days_since_start = (date - dates[0]).days
        years_since_start = days_since_start / 365.25
        day_growth = self.base_mean * ((1 +  self.annual_growth_rates) ** years_since_start)

        noise = np.random.normal(loc=0.0, scale=0.1 * self.base_mean)  
        
        sales = day_growth * weekday_seasonality * annual_seasonality * (1 + promo_lift) * competitor_multiplier * holiday_effect + noise
        
        sales = max(0.0, sales)
        
        day_info = {
                    "date": date,
                    "store_id": self.city,
                    "daily_sales": round(sales, 2),
                    "promotion_active": int(promo_flag),
                    "store_size": self.size,
                    "city_population": self.population,
                    "day_of_week": date.weekday(),
                    "is_weekend": int(date.weekday() >= 5),
                    "holiday_flag": holiday_flag,  
                    "school_holiday_flag": 1 if date.normalize() in self.school_holidays else 0,  
                    "year": date.year,
                    "month": date.month,
                    "competitor_opened_flag": competitor_flag,
                    "competitor_impact": round(competitor_multiplier, 3),
                    "base_mean": round(self.base_mean, 2),
                    "annual_growth_rate": round(self.annual_growth_rates, 4),
                    }

        return day_info


    def create_df(self):
        records = []
        for date in self.dates:
            day_info = self.daily_income(date)
            records.append(day_info)
        self.df = pd.DataFrame.from_records(records)

    def show_salary(self):
        print("Stores:", self.city)
        print("Competitor impacted store:", self.competitor_impact)
        if self.competitor_open_date:
            print("Competitor opened on:", self.competitor_open_date.date(), "impact:", round(self.competitor_impact, 3))
        print(self.df.head(10))
        print(self.df.tail(5))

        print("\nSummary by store (first/last dates and average daily sales):")
        summary = self.df.groupby("store_id").agg(first_date=("date", "min"),
                                             last_date=("date", "max"),
                                             avg_daily_sales=("daily_sales", "mean"),
                                             total_days=("date", "count"),
                                             avg_promo_days=("promotion_active", "mean"))
        print(summary)

        plt.figure(figsize=(12,4))
        subset = self.df.set_index("date")
        subset["daily_sales"].rolling(14, center=True).mean().plot(label=f"{self.city} 14-day MA")
        subset["daily_sales"].plot(alpha=0.3, label="daily_sales")
        if self.competitor_open_date:
            plt.axvline(self.competitor_open_date, color="red", linestyle="--", linewidth=1.5, label="Competitor opened")
            ymax = self.df["daily_sales"].max()

            plt.annotate(
            f"Competitor opened\n{self.competitor_open_date.date()}",
            xy=(self.competitor_open_date, ymax),
            xytext=(10, -10),
            textcoords="offset points",
            rotation=90,
            va="top",
            color="red",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
            )

        plt.title(f"Daily sales for {self.city} with {self.size} size")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{self.city}.png")

        
if __name__=='__main__':
    print('Start')
    cities = {
                "Tehran": 8846782,
                "Isfahan": 1570860,
                "Shiraz": 1229411,
                "Tabriz": 1582068,
                "Urumieh": 1265547,
                "Mashhad": 3540000,
                "Ahvaz": 1350000,
                "Karaj": 262949,
                "Ardabil": 606000,
                "Bushehr": 231000,
                "Khoram Ababd": 133000,
                "Birjand": 187000,
                "Bojnord": 203000,
                "Golestan": 389000,
                "Bandar Abbas": 680000,
                "Ilam": 197000,
                "Kerman": 537000,
                "Kermanshah": 501000,
                "Kurdistan": 545000,
                "Arak": 373000,
                "Sari": 524000,
                "Qazvin": 402748,
                "Qom": 1312061,
                "Zahedan": 672000,
                "Semnan": 199000,
                "Yazd": 529673,
                "Zanjan": 429000
            }
    n_stores = 5
    start_date = "2022-01-01"   
    end_date = "2024-12-31"
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    store_sizes = random.choices(["small", "medium", "large"], k=n_stores)
    city_choises = random.choices(list(cities.keys()), k=n_stores)
    city_populations = [random.randint(0, len(cities)) for _ in range(n_stores)] 

    competitor_open_date = pd.Timestamp("2023-06-15")
    competitor_store = np.random.choice(city_choises)

    stores = dict()
    dfs = []
    for i, num_store in enumerate(range(n_stores)):
        city = city_choises[num_store]
        population = cities[city]
        competitor_date = competitor_open_date if (city == competitor_store) else None
        stores[num_store] = InnovateMart(store_city=city + str(i), store_size=store_sizes[num_store], city_population=population, dates=dates, competitor_open_date=competitor_date)
        stores[num_store].create_df()
        dfs.append(stores[num_store].df)
        stores[num_store].show_salary()
    df = pd.concat(dfs, ignore_index=True)

    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)

    df_long = df.copy()
    df_long = df_long.rename(columns={"date": "Date"})

    print("\nSample rows:")
    print(df_long[["Date","store_id","daily_sales","promotion_active","store_size","city_population","is_weekend","competitor_opened_flag"]].head(12))
    df_long.to_csv("simulated_innovatemart_daily_sales_2.csv", index=False)
    print('done')
