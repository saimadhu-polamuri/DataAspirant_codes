"""
===============================================
Objective: Implementing Markov Chains model
Author: Venkatesh Nagilla
Blog: https://dataaspirant.com
Date: 2020-08-09
===============================================
"""

import pymc3 as pm
model = pm.Model()

import pymc3.distributions.continuous as pmc
import pymc3.distributions.discrete as pmd
import pymc3.math as pmm

with model:
    passenger_onboarding = pmc.Wald('Passenger Onboarding', mu=0.5, lam=0.2)
    refueling = pmc.Wald('Refueling', mu=0.25, lam=0.5)
    departure_traffic_delay = pmc.Wald('Departure Traffic Delay', mu=0.1, lam=0.2)

    departure_time = pm.Deterministic('Departure Time',
                                      12.0 + departure_traffic_delay +
                                      pmm.switch(passenger_onboarding >= refueling,
                                                 passenger_onboarding,
                                                 refueling))

    rough_weather = pmd.Bernoulli('Rough Weather', p=0.35)

    flight_time = pmc.Exponential('Flight Time', lam=0.5 - (0.1 * rough_weather))
    arrival_traffic_delay = pmc.Wald('Arrival Traffic Delay', mu=0.1, lam=0.2)

    arrival_time = pm.Deterministic('Arrival time',
                                    departure_time +
                                    flight_time +
                                    arrival_traffic_delay)


nb_samples = 500
with model:
    samples = pm.sample(draws=nb_samples, random_seed=1000)
