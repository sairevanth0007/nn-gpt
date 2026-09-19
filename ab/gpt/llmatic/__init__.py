"""LLMatic-style MAP-Elites quality-diversity layer for the iterative pipeline.

This package is additive: it never modifies the core pipeline in place. It is
wired into the generation subprocess via ``patches/usercustomize.py`` (see
``patches/`` for the injection point). See ``archive.py`` for the MAP-Elites
archive, ``descriptors.py`` for pre-eval behavioral-descriptor extraction, and
``prompts.py`` for the mutation / crossover prompt templates.
"""
