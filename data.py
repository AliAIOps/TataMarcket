import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt

RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)

class InnovateMart:


def init(self, store_city, store_size, city_population):

Store metadata
self.city = store_city
self.size = store_size
self.population = city_population

Map discrete size categories to numeric multipliers.
size_factor = {"small": 1.0, "medium": 1.6, "large": 3.0}[store_size]

Compute a heuristic base daily-sales mean:
1000 * size_factor scales with store size,
(1 + 0.1 * city_population) introduces dependence on city population.
Note: city_population is expected to be a (possibly large) integer; this
formula will produce very large values if population is the full city population.
self.base_mean = 1000 * size_factor * (1 + 0.1 * city_population) # hypothetical base daily sales

def promotion(self):
"""
Placeholder method for modeling promotions.

Should return a multiplicative or additive effect on base_mean depending
on active promotions (for example: 1.2 for +20% uplift).
"""
pass

def seasonality(self):
"""
Placeholder method for modeling seasonal effects.

Could return a factor or array of factors per date representing weekly
or yearly seasonality to modulate sales.
"""
pass

def competitor_store(self):
"""
Placeholder method to model competitor impact.

Could return a reduction factor or noise term representing nearby
competing stores opening/closing or promotional actions.
"""
pass

if name == 'main':

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
    def __init__(self, store_city, store_size, city_population):
        self.city = store_city
        self.size = store_size
        self.population = city_population 
        size_factor = {"small": 1.0, "medium": 1.6, "large": 3.0}[store_size]
        self.base_mean = 1000 * size_factor * (1 + 0.1 * city_population)  # عددی فرضی برای فروش پایه روزانه

    def promotion(self):
        pass
    
    def seasonality(self):
        pass

    def competitor_store(self):
        pass

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
    # پارامترهای کلی شبیه‌سازی
    n_stores = 5                # می‌توانید بین 3 تا 5 انتخاب کنید
    start_date = "2022-01-01"   # شروع بازه
    end_date = "2024-12-31"     # پایان بازه (تقریباً 3 سال)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    # برای تنوع، اندازه‌ها را پخش می‌کنیم و city_population را به صورت تقریباً واقعی می‌سازیم
    store_sizes = random.choices(["small", "medium", "large"], k=n_stores)
    city_choises = random.choices(list(cities.keys()), k=n_stores)
    city_populations = [random.randint(0, len(cities)) for _ in range(n_stores)]  # عدد جمعیت شهری برای هر فروشگاه
    stores = dict()
    for num_store in range(n_stores):
        city = city_choises[num_store]
        population = cities[city]
        stores[num_store] = InnovateMart(store_city=city, store_size=store_sizes[num_store], city_population=population, )
    print('done')
