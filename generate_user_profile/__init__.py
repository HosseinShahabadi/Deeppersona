"""DeepPersona user-profile generation package.

The scripts in this package are designed to be run directly, e.g.:

    python generate_user_profile/generate_profile.py --num-profiles 50 --attribute-count 200

They use absolute imports (``from config import ...``) and therefore expect
to be executed from within this directory (which puts it on ``sys.path``).
This ``__init__.py`` is intentionally minimal; it exists only so the folder
is a valid Python package.
"""

__all__ = []
