# Contributing to NetSentinel-RL

## Setup
```bash
conda env create -f environment.yml
conda activate netsentinel
pre-commit install
```

## Running Tests
```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

## Notebook Conventions
- Run notebooks in order (01 → 08)
- Clear outputs before committing: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
- Each notebook must be self-contained (runs from top to bottom without errors)

## Code Style
- Black formatting: `black .`
- Type hints on all public functions
- Docstrings: Google style

## Adding a New Agent
1. Subclass `agents/base_agent.py:BaseAgent`
2. Implement `predict()` returning `AgentDecision`
3. Add environment action space in `environment/network_env.py`
4. Register in `orchestrator/task_router.py`
5. Add tests in `tests/test_agents.py`
