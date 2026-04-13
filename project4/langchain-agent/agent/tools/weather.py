"""
Weather tool using Open-Meteo API — completely free, no API key required.

CONCEPT: This shows how tools can call external APIs without credentials.
The agent doesn't know HOW this works — it just knows what the tool does
from the docstring. Good docstrings are critical for agent tool selection.
"""

import requests
from langchain_core.tools import tool


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# WMO Weather interpretation codes
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Light rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Light showers",
    81: "Moderate showers",
    82: "Heavy showers",
    95: "Thunderstorm",
}


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for any city in the world.
    Use this when the user asks about weather, temperature, or conditions in a location.
    Input should be a city name like 'London' or 'New York' or 'Karachi'.
    """
    try:
        # Step 1: Geocode the city name to coordinates
        geo_response = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 1},
            timeout=10
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find city: {city}"

        location = geo_data["results"][0]
        lat = location["latitude"]
        lng = location["longitude"]
        city_name = location.get("name", city)
        country = location.get("country", "")

        # Step 2: Fetch current weather
        weather_response = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lng,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "weather_code"
                ],
                "temperature_unit": "celsius"
            },
            timeout=10
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")
        code = current.get("weather_code", 0)
        conditions = WMO_CODES.get(code, "Unknown")

        return (
            f"Weather in {city_name}, {country}:\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind} km/h\n"
            f"Conditions: {conditions}"
        )

    except requests.RequestException as e:
        return f"Weather unavailable: {str(e)}"

    except Exception as e:
        return f"Weather error: {str(e)}"
