def __getattr__(name):
    if name == "Session":
        from kwave.cli.session import Session

        return Session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Session"]
