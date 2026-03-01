"""Shared tracking modules for IC, disagreement, and execution quality metrics.

Decouples metric persistence from the API layer so that autopilot and other
subsystems can record tracking data without importing api.services.

This package has NO dependency on ``api/`` or ``autopilot/`` —
it only depends on ``config`` and the standard library.
"""
