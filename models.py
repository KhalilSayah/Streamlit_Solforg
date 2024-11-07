from datetime import date, datetime
from typing import List, Literal, Union
from pydantic import BaseModel

# Model init


class FinanceRound(BaseModel):
    label: str
    start: date
    end: date
    raised_funds: int
    valuation: int

class FinancingRounds(BaseModel):
    rounds: List[FinanceRound]

class ModelInit(BaseModel):
    max_supply: int
    listing_price: float
    base_alloc: float
    bonus_alloc: float
    employees: int
    finance_rounds: FinancingRounds

    def get_btu(self, alloc="base"):
        alloc_value = self.bonus_alloc if alloc == "bonus" else self.base_alloc
        total_token = self.max_supply * alloc_value
        return int(total_token / self.employees)


# Employee Distribution
class NumericCriteriaPart(BaseModel):
    label: str
    min_value: float
    max_value: float
    score: float

# Define a class for categorical-based criteria parts
class CategoricalCriteriaPart(BaseModel):
    label: str
    score: float

# Define a class for criteria that is numeric-based with score calculation logic
class NumericCriteria(BaseModel):
    criteria_type : Literal['bonus', 'primary']
    label: str
    criteria_parts: List[NumericCriteriaPart]

    # Method to get score based on a numeric input
    def get_score(self, value: float) -> float:
        for part in self.criteria_parts:
            if part.min_value <= value < part.max_value:
                return part.score
        return 0  # Default score if no range is matched

# Define a class for criteria that is categorical-based with score calculation logic
class CategoricalCriteria(BaseModel):
    criteria_type : Literal['bonus', 'primary']
    label: str
    criteria_parts: List[CategoricalCriteriaPart]

    # Method to get score based on a category input
    def get_score(self, category: str) -> float:
        for part in self.criteria_parts:
            if part.label == category:
                return part.score
        return 0  # Default score if no category is matched

# Define a general Criteria type that can be either numeric or categorical
Criteria = Union[NumericCriteria, CategoricalCriteria]
    

