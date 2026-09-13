"""Typed view over the Nemotron-Personas-Brazil columns we use.

Six target sections, fixed order (ADR 0004). The order is load-bearing: evaluation
scores section presence *and* ordering, and the model is trained to reproduce it.
"""

from __future__ import annotations

from dataclasses import dataclass

SECTIONS: tuple[tuple[str, str], ...] = (
    ("Síntese", "persona"),
    ("Perfil profissional", "professional_persona"),
    ("Origem cultural", "cultural_background"),
    ("Habilidades", "skills_and_expertise"),
    ("Hobbies e interesses", "hobbies_and_interests"),
    ("Objetivos de carreira", "career_goals_and_ambitions"),
)

# Narrative columns consulted when extracting the name (ADR 0005). The lifestyle facets
# are not training targets, but they are extra confirmation signal for mononyms.
NAME_SOURCE_COLUMNS: tuple[str, ...] = (
    "professional_persona",
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
)

ATTRIBUTE_COLUMNS: tuple[str, ...] = (
    "sex",
    "age",
    "marital_status",
    "education_level",
    "occupation",
    "municipality",
    "state",
    "country",
)


@dataclass(frozen=True)
class PersonaAttributes:
    name: str
    sex: str
    age: int
    marital_status: str
    education_level: str
    occupation: str
    municipality: str
    state: str
    country: str

    @classmethod
    def from_row(cls, row: dict, name: str) -> "PersonaAttributes":
        return cls(
            name=name,
            sex=row["sex"],
            age=int(row["age"]),
            marital_status=row["marital_status"],
            education_level=row["education_level"],
            occupation=row["occupation"],
            municipality=row["municipality"],
            state=row["state"],
            country=row["country"],
        )
