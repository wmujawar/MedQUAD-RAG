from typing import Any, Callable


class Container:
    def __init__(self) -> None:
        self._providers: dict[str, tuple[Callable[[], Any], bool]] = {}
        self._singletons: dict[str, Any] = {}

    def register(
        self, name: str, provider: Callable[[], Any], singleton: bool = False
    ) -> None:
        self._providers[name] = (provider, singleton)

    def resolve(self, name: str) -> Any:
        if name in self._singletons:
            return self._singletons[name]

        if name not in self._providers:
            raise ValueError(f"No provider registered for {name}")

        provider, singleton = self._providers[name]
        instance = provider()

        if singleton:
            self._singletons[name] = instance

        return instance
