from __future__ import annotations
from typing import Generic, TypeVar, Callable, List

I = TypeVar("I")
O = TypeVar("O")

class Runnable(Generic[I, O]):
    def invoke(self, inp: I) -> O:
        raise NotImplementedError

    def __or__(self, other: Runnable[O, any]):
        return RunnableSequence(self, other)

    def __ror__(self, other: Runnable[any, I]):
        return RunnableSequence(other, self)


class RunnableLambda(Runnable[I, O]):
    def __init__(self, fn: Callable[[I], O]):
        self.fn = fn

    def invoke(self, inp: I) -> O:
        return self.fn(inp)


class RunnableSequence(Runnable[I, O]):
    def __init__(self, *steps: Runnable):
        self.steps: List[Runnable] = list(steps)

    def invoke(self, inp: I) -> O:
        out = inp
        for step in self.steps:
            out = step.invoke(out)
        return out
