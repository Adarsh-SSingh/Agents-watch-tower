import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

TIMEOUT = 10


def get_weather(city: str, country: str = None):
    if not city or not city.strip():
        raise ValueError("City name cannot be empty.")

    city = city.strip()
    country = country.strip() if country else None

    try:
        # 1. Geocode the requested city
        geo_response = requests.get(
            GEOCODING_URL,
            params={
                "name": city,
                "count": 10,
                "language": "en",
                "format": "json"
            },
            timeout=TIMEOUT
        )
        geo_response.raise_for_status()

        locations = geo_response.json().get("results", [])

        if not locations:
            raise ValueError(f"Location not found: {city}")

        # 2. Filter by country if user specified one
        if country:
            country_lower = country.lower()

            matching_locations = [
                loc for loc in locations
                if (
                    loc.get("country", "").lower() == country_lower
                    or loc.get("country_code", "").lower() == country_lower
                )
            ]

            if not matching_locations:
                raise ValueError(
                    f"{city} was not found in {country}."
                )

            locations = matching_locations

        # 3. Avoid silently choosing an ambiguous location
        if len(locations) > 1:
            candidates = [
                {
                    "city": loc.get("name"),
                    "country": loc.get("country"),
                    "country_code": loc.get("country_code"),
                    "admin1": loc.get("admin1")
                }
                for loc in locations[:5]
            ]

            countries = {
                loc.get("country_code")
                for loc in locations
            }

            if len(countries) > 1:
                return {
                    "status": "ambiguous",
                    "requested_city": city,
                    "message": (
                        "Multiple locations found. "
                        "Please specify the country."
                    ),
                    "candidates": candidates
                }

        location = locations[0]

        # 4. Fetch current weather
        weather_response = requests.get(
            WEATHER_URL,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "precipitation,"
                    "weather_code,"
                    "cloud_cover,"
                    "wind_speed_10m,"
                    "wind_direction_10m"
                ),
                "timezone": "auto"
            },
            timeout=TIMEOUT
        )
        weather_response.raise_for_status()

        data = weather_response.json()
        current = data.get("current")

        if not current:
            raise RuntimeError("Weather data unavailable.")

        return {
            "status": "success",
            "location": {
                "city": location.get("name"),
                "country": location.get("country"),
                "country_code": location.get("country_code"),
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "timezone": data.get("timezone")
            },
            "weather": {
                "temperature_c": current.get("temperature_2m"),
                "feels_like_c": current.get("apparent_temperature"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation"),
                "weather_code": current.get("weather_code"),
                "cloud_cover_percent": current.get("cloud_cover"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "wind_direction_degrees": current.get("wind_direction_10m"),
                "observed_at": current.get("time")
            }
        }

    except requests.Timeout as exc:
        raise RuntimeError("Weather API request timed out.") from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Weather API request failed: {exc}"
        ) from exc