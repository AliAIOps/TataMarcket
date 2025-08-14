import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import argparse
from config import cities, public_cfg

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)

class InnovateMart:
    """
    Represent a single InnovateMart store located in a city.

    Attributes
    ----------
    city : str
        The city name where the store is located.
    size : str
        Store size category; expected values: "small", "medium", "large".
    population : int
        Population of the city (used as a factor in sales calculation).
    base_mean : float
        A heuristic estimate of the store's base daily sales. Calculated from
        store size and city population to provide a baseline for simulation.
    start_date : pd.Timestamp
        Simulation start date.
    end_date : pd.Timestamp
        Simulation end date.
    competitor_open_date : pd.Timestamp or None
        Date competitor store opened near this store.
    Methods
    -------
    promotion()
        Generate dates when promotions are active.
    holiday()
        Generate holiday and school holiday dates.
    seasonality(date)
        Calculate weekly and annual seasonality factors for a given date.
    daily_income(date)
        Calculate simulated daily sales based on all factors.
    create_df()
        Create a DataFrame of simulated daily sales for all dates.
    show_salary()
        Display sample data and plot sales trend.
    """

    def __init__(self, store_city, store_size, city_population, start_date, end_date, competitor_open_date=None):
        # Store basic attributes
        self.city = store_city
        self.size = store_size
        self.population = city_population 
        #Dates for which sales data will be simulated.
        self.dates = pd.date_range(start=start_date, end=end_date, freq="D")
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # Size factor affects base mean sales: larger stores sell more
        size_factor = {"small": 1.0, "medium": 1.6, "large": 3.0}[store_size]
        
        # Base daily sales heuristic using store size and city population
        self.base_mean = 100 * size_factor * (1 + 0.1 * city_population)
        
        # Annual growth rate: random mild growth between 2% to 8% per year
        self.annual_growth_rates = np.random.uniform(0.02, 0.08)
        
        # Competitor opening date and impact factor (12% to 30% drop in sales)
        self.competitor_open_date = competitor_open_date
        self.competitor_impact = np.random.uniform(0.12, 0.30)

        # Generate promotion dates (active days with sales uplift)
        self.promo_dates = self.promotion()
        
        # Define holiday and school holiday date sets
        self.school_holidays, self.holiday_dates = self.holiday()

    def promotion(self):
        """
        Generate promotion active dates within the simulation period.
        Promotions can increase sales by 20% to 70% depending on store size.
        """
        # Number of promotional campaigns per year between 8 and 15
        campaigns_per_year = np.random.randint(8, 16)
        
        # All possible dates in the simulation range
        all_possible = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        
        # Randomly choose campaign start dates for each year
        chosen = np.random.choice(
            all_possible, 
            size=campaigns_per_year * ((self.end_date.year - self.start_date.year) + 1), 
            replace=False
        )
        
        promo_dates = []
        for d in chosen:
            # Each campaign lasts between 1 and 4 days
            length = np.random.randint(1, 5)
            rng = pd.date_range(start=d, periods=length, freq="D")
            promo_dates.extend(list(rng))
        
        # Keep only unique dates within the simulation period
        promo_dates = sorted(list({pd.Timestamp(d).normalize() for d in promo_dates if (self.start_date <= pd.Timestamp(d) <= self.end_date)}))
        
        return promo_dates

    def holiday(self):
        """
        Define fixed holiday dates and school holiday periods.
        Holidays can increase sales by 50%, school holidays by 15%.
        """
        # Predefined set of key holidays (sample dates)
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
        
        school_holidays = set()
        
        # Add school holidays: summer break and winter break each year
        for year in range(self.start_date.year, self.end_date.year + 1):
            # Summer holiday assumed Tir 1 to Shahrivar 30
            summer_start = pd.to_datetime(f"{year}-04-01")
            summer_end = pd.to_datetime(f"{year}-06-30")
            for d in pd.date_range(summer_start, summer_end):
                school_holidays.add(d.normalize())
            
            # Winter break for students in university: Dey 24 to Bahman 24
            for d in pd.date_range(f"{year}-10-24", f"{year}-11-24"):
                school_holidays.add(d.normalize())
        
        return school_holidays, holiday_dates

    def seasonality(self, date):
        """
        Calculate weekly and annual seasonality multipliers.
        Weekends have 25% higher sales; winter/fall months have 30% higher sales.
        """
        dow = date.weekday()  # Saturday=0,...,Friday=6
        
        # Weekend and Friday uplift
        if dow >= 5:
            weekday_seasonality = 1.25  # 25% higher sales on Thursday, Friday
        else:
            weekday_seasonality = 1.0   # Normal weekdays
        
        month = date.month
        
        # Seasonal uplift in some months like a new year(Esfand and Farvardin) and beging schools/universities days (Shahrivar, Mehr)in Iran  
        if month in [6, 7, 1, 12]:
            annual_seasonality = 1.30  # 30% uplift in winter/fall months
        else:
            annual_seasonality = 1.00  # Normal months
        
        return weekday_seasonality, annual_seasonality

    def daily_income(self, date):
        """
        Calculate daily sales amount for a given date including all effects:
        seasonality, promotions, holidays, competitor impact, and random noise.
        """
        # Get seasonality multipliers
        weekday_seasonality, annual_seasonality = self.seasonality(date)
        
        # Check if promotion is active
        promo_flag = 1 if date in self.promo_dates else 0
        
        # Promotion effect: increases sales by 20% to 70% scaled by store size
        if promo_flag:
            promo_lift = np.random.uniform(0.20, 0.70) * (1.0 if self.size == "small" else (0.9 if self.size=="medium" else 0.8))
        else:
            promo_lift = 0.0

        # Holiday effects
        holiday_flag = 1 if date.normalize() in self.holiday_dates else 0
        if holiday_flag:
            holiday_effect = 1.5       # 50% increase on holidays
        elif date.normalize() in self.school_holidays:
            holiday_effect = 1.15      # 15% increase during school holidays
        else:
            holiday_effect = 1.0

        # Competitor effect: after competitor opens, sales drop by 12%-30%
        competitor_flag = 1 if (self.competitor_open_date is not None and date >= self.competitor_open_date) else 0
        competitor_multiplier = (1 - self.competitor_impact) if competitor_flag else 1.0

        # Calculate growth adjusted base sales (compound annual growth)
        days_since_start = (date - self.dates[0]).days
        years_since_start = days_since_start / 365.25
        day_growth = self.base_mean * ((1 + self.annual_growth_rates) ** years_since_start)

        # Add random Gaussian noise (12% of base mean)
        noise = np.random.normal(loc=0.0, scale=0.05 * self.base_mean)

        # Combine all factors to get final sales number for the day
        sales = day_growth * weekday_seasonality * annual_seasonality * (1 + promo_lift) * competitor_multiplier * holiday_effect + noise
        
        # Prevent negative sales
        sales = max(0.0, sales)

        # Prepare dictionary with daily info and metadata
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
        """
        Create the full DataFrame with daily sales data for the store by simulating
        each day within the date range.
        """
        records = []
        for date in self.dates:
            day_info = self.daily_income(date)
            records.append(day_info)
        self.df = pd.DataFrame.from_records(records)

    def show_salary(self, add='./'):
        """
        Print some sample rows and summary statistics, then plot daily sales trend.
        If competitor opened, show vertical line indicating the date.
        """
        print("Stores:", self.city)
        print("Competitor impacted store:", self.competitor_impact)
        if self.competitor_open_date:
            print("Competitor opened on:", self.competitor_open_date.date(), "impact:", round(self.competitor_impact, 3))
        
        print(self.df.head(10))
        print(self.df.tail(5))
        
        print("\nSummary by store (first/last dates and average daily sales):")
        summary = self.df.groupby("store_id").agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            avg_daily_sales=("daily_sales", "mean"),
            total_days=("date", "count"),
            avg_promo_days=("promotion_active", "mean")
        )
        print(summary)

        # Plot 14-day moving average and daily sales
        plt.figure(figsize=(12,4))
        subset = self.df.set_index("date")
        subset["daily_sales"].rolling(14, center=True).mean().plot(label=f"{self.city} 14-day MA")
        subset["daily_sales"].plot(alpha=0.3, label="daily_sales")
        
        # Mark competitor opening date if any
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
        plt.savefig(f"{add}/{self.city}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Simulate InnovateMart sales data")
    parser.add_argument("--n_stores", type=int, default=public_cfg["NUM_STORES"], help="Number of stores to simulate (default: 5)")
    parser.add_argument("--start_date", type=str, default=public_cfg["START_DATE"], help="Start date for simulation (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=public_cfg["END_DATE"], help="End date for simulation (YYYY-MM-DD)")
    parser.add_argument("--competitor_date", type=str, default=public_cfg["COMPETITOR_DATE"], help="Competitor opening date (YYYY-MM-DD)")
    parser.add_argument("--save", type=str, default=public_cfg['DATA_PATH'], help="Path of save csvs and images")

    args = parser.parse_args()
    n_stores = args.n_stores
    start_date = args.start_date
    end_date = args.end_date

    #create save folder
    os.makedirs(args.save, exist_ok=True)

    print('Start simulation')

    # Population dictionary for cities
    # Randomly assign store sizes and cities for simulation
    store_sizes = random.choices(["small", "medium", "large"], k=n_stores)
    city_choices = random.choices(list(cities.keys()), k=n_stores)

    # Competitor opening date and a randomly selected competitor store city
    competitor_open_date = pd.Timestamp(args.competitor_date)
    competitor_store = np.random.choice(city_choices)

    stores = dict()
    dfs = []

    # Create InnovateMart store instances, generate data and display summaries
    for i in range(n_stores):
        city = city_choices[i]
        population = cities[city]
        competitor_date = competitor_open_date if (city == competitor_store) else None
        stores[i] = InnovateMart(
            store_city=city + str(i),
            store_size=store_sizes[i],
            city_population=population,
            start_date=start_date,
            end_date=end_date,
            competitor_open_date=competitor_date
        )
        stores[i].create_df()
        dfs.append(stores[i].df)
        stores[i].show_salary(args.save)

    # Combine data from all stores into one DataFrame
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)

    # Rename 'date' column to 'Date' for user convenience
    df_long = df.copy()
    df_long = df_long.rename(columns={"date": "Date"})

    print("\nSample rows:")
    print(df_long[["Date","store_id","daily_sales","promotion_active","store_size","city_population","is_weekend","competitor_opened_flag"]].head(12))

    # Save the simulated data to CSV
    df_long.to_csv(f"{args.save}/simulated_innovatemart_daily_sales.csv", index=False)
    
    print('Simulation done')