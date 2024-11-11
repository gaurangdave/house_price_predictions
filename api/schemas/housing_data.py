from pydantic import BaseModel, Field
from typing import Optional


class HousingData(BaseModel):
    longitude: float = Field(..., description="Longitude of the location")
    latitude: float = Field(..., description="Latitude of the location")
    housing_median_age: float = Field(...,
                                      description="Median age of the houses in the area")
    total_rooms: float = Field(...,
                               description="Total number of rooms in all houses in the area")
    total_bedrooms: Optional[float] = Field(
        None, description="Total number of bedrooms in all houses in the area")
    population: float = Field(..., description="Population in the area")
    households: float = Field(...,
                              description="Number of households in the area")
    median_income: float = Field(..., description="Median income in the area")
    ocean_proximity: str = Field(
        ..., description="Proximity to the ocean (e.g., 'INLAND', '<1H OCEAN', etc.)")
    income_categories: int = Field(...,
                                   description="Income category of the area")
    population_categories: int = Field(...,
                                       description="Population category of the area")

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -118.4,
                "latitude": 34.1,
                "housing_median_age": 28.0,
                "total_rooms": 700.0,
                "total_bedrooms": 300.0,
                "population": 1200.0,
                "households": 400.0,
                "median_income": 5.0,
                "ocean_proximity": "INLAND",
                "income_categories": 2,
                "population_categories": 1
            }
        }
