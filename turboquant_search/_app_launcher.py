"""
Bridge between CLI and Gradio app.
Allows `tqs demo` to pass pre-loaded data into the dashboard.
"""

import numpy as np
from typing import List, Optional


# Module-level state set by CLI before app import
_demo_vectors: Optional[np.ndarray] = None
_demo_queries: Optional[np.ndarray] = None
_demo_texts: Optional[List[str]] = None
_demo_info: Optional[dict] = None


def launch_demo(
    vectors: np.ndarray,
    queries: np.ndarray,
    texts: List[str],
    info: dict,
    port: int = 7860,
    share: bool = False,
):
    """Set demo data and launch the Gradio app."""
    global _demo_vectors, _demo_queries, _demo_texts, _demo_info
    _demo_vectors = vectors
    _demo_queries = queries
    _demo_texts = texts
    _demo_info = info

    # Import app module (which reads these globals) and launch
    import importlib
    import sys
    import os

    # Add project root to path if needed
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import and launch app.py
    from app import demo as gradio_demo
    gradio_demo.launch(server_port=port, share=share, show_error=True)


def get_demo_data():
    """Retrieve pre-loaded demo data (or None if launched directly)."""
    return _demo_vectors, _demo_queries, _demo_texts, _demo_info
