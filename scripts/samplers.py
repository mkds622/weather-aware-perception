import numpy as np
import random

# ---- RANDOM (current) ----
def random_sampler(regime):
    if regime == "fog":
        return {
            "cloudiness": random.uniform(50, 100),
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "wind_intensity": random.uniform(0, 20),
            "sun_altitude_angle": random.uniform(10, 40),
            "fog_density": random.uniform(30, 90),
            "fog_distance": random.uniform(5, 50),
            "fog_falloff": random.uniform(0.5, 2.0),
            "wetness": 0.0
        }

    if regime == "rain":
        return {
            "cloudiness": random.uniform(60, 100),
            "precipitation": random.uniform(30, 100),
            "precipitation_deposits": random.uniform(30, 100),
            "wind_intensity": random.uniform(10, 50),
            "sun_altitude_angle": random.uniform(5, 30),
            "fog_density": random.uniform(0, 20),
            "fog_distance": random.uniform(50, 100),
            "fog_falloff": 1.0,
            "wetness": random.uniform(30, 100)
        }

    return {
        "cloudiness": random.uniform(0, 30),
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": random.uniform(0, 10),
        "sun_altitude_angle": random.uniform(30, 80),
        "fog_density": 0.0,
        "fog_distance": 100.0,
        "fog_falloff": 1.0,
        "wetness": 0.0
    }


# ---- LHS (Latin Hypercube Sampling) ----
def latin_hypercube(n, d):
    result = np.zeros((n, d))
    for i in range(d):
        perm = np.random.permutation(n)
        result[:, i] = (perm + np.random.rand(n)) / n
    return result


def lhs_sampler(regime, n_samples):
    # define active dimensions per regime
    if regime == "fog":
        # [cloudiness, wind, sun_alt, fog_density, fog_distance, fog_falloff]
        lhs = latin_hypercube(n_samples, 6)
        samples = []

        for row in lhs:
            c, wind, sun, fd, fdist, ff = row

            samples.append({
                "cloudiness": 50 + c * 50,
                "precipitation": 0.0,
                "precipitation_deposits": 0.0,
                "wind_intensity": wind * 20,
                "sun_altitude_angle": 10 + sun * 30,
                "fog_density": 30 + fd * 60,
                "fog_distance": 5 + (1 - fd) * 45,
                "fog_falloff": 0.5 + ff * 1.5,
                "wetness": 0.0
            })

        return samples

    if regime == "rain":
        # [cloudiness, precipitation, wind, sun_alt, wetness]
        lhs = latin_hypercube(n_samples, 5)
        samples = []

        for row in lhs:
            c, p, wind, sun, w = row

            samples.append({
                "cloudiness": 60 + c * 40,
                "precipitation": 30 + p * 70,
                "precipitation_deposits": 30 + p * 70,
                "wind_intensity": 10 + wind * 40,
                "sun_altitude_angle": 5 + sun * 25,
                "fog_density": 0.0,
                "fog_distance": 100.0,
                "fog_falloff": 1.0,
                "wetness": 30 + w * 70
            })

        return samples

    # clear
    lhs = latin_hypercube(n_samples, 3)
    samples = []

    for row in lhs:
        c, wind, sun = row

        samples.append({
            "cloudiness": c * 30,
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "wind_intensity": wind * 10,
            "sun_altitude_angle": 30 + sun * 50,
            "fog_density": 0.0,
            "fog_distance": 100.0,
            "fog_falloff": 1.0,
            "wetness": 0.0
        })

    return samples