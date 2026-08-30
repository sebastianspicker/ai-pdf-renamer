"""Low-level infrastructure primitives shared by Folionym subsystems.

This package deliberately depends only on the standard library and external
transport libraries. Application, extraction, naming, LLM, and interface code
may depend on it; infrastructure never depends on those higher layers.
"""
