import typing
import numpy as np

import pydantic

T = typing.TypeVar("T", bound=np.generic)


class NDArray(np.ndarray, typing.Generic[T]):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls, data: typing.Any, field: pydantic.fields.ModelField | None = None
    ) -> "NDArray":
        return np.asarray(data, dtype=np.float32)  # type: ignore
