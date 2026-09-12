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
