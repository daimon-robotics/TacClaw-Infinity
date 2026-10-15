"""User-facing exceptions for the replay-only application."""


class ReplayError(RuntimeError):
    """A configuration, data, model, or replay error safe to show to a user."""
