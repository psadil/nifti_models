import pathlib
import re

import pydantic

from . import fields


def extract_entity(entity: str, source: str) -> str | None:
    if found := re.search(rf"(?<={entity}-)[\w]+(?=_)", source):
        return found.group(0)


class Entities(pydantic.BaseModel):
    sub: str | None = None
    ses: str | None = None
    task: str | None = None
    acq: str | None = None
    rec: str | None = None
    dir: str | None = None
    run: str | None = None
    echo: str | None = None
    part: fields.Part | None = None
    suffix: fields.Suffix | None = None

    class Config:
        frozen = True

    @classmethod
    def from_path(cls, path: pathlib.Path):
        return cls(**parse_entities(path))


def parse_entities(path: pathlib.Path) -> dict[str, str]:
    if suffix := re.search(rf"(?<=_)({fields.SUFFIXES})(?=.)", path.name):
        entities = {"suffix": suffix.group(0)}
        for entity in [e for e in Entities.schema().get("properties", {}).keys()]:
            if extracted_value := extract_entity(entity, path.name):
                entities.update({entity: extracted_value})
        return entities
    else:
        raise AssertionError("not sure how to parse entities from this filename")
