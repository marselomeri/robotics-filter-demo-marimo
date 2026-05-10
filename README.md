# 2D Filter Comparison in marimo

This project is a small interactive simulation for teaching the difference between:

- a **Markov motion-propagation approach**
- a **Bayesian histogram filter**
- a **Kalman filter**

The scenario is a simple 2D quadcopter-like motion from point `A` to point `B`.

## Local run

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install marimo matplotlib numpy pandas
marimo edit filter_demo_app.py
```

Or run it directly as an app:

```bash
marimo run filter_demo_app.py
```

## Open from GitHub in mo lab

marimo's official docs say notebooks hosted on GitHub can be opened by appending the GitHub file URL to `molab.marimo.io`.

This project is now published here:

- Repo: `https://github.com/marselomeri/robotics-filter-demo-marimo`
- Notebook file: `https://github.com/marselomeri/robotics-filter-demo-marimo/blob/main/filter_demo_app.py`
- Open directly in mo lab: `https://molab.marimo.io/github/marselomeri/robotics-filter-demo-marimo/blob/main/filter_demo_app.py`

In general, the pattern is:

```text
https://molab.marimo.io/github/<owner>/<repo>/blob/main/filter_demo_app.py
```

Example format from the official marimo page:

```text
https://molab.marimo.io/github/owner/repo/blob/main/notebook.py
```

Official references:

- https://molab.marimo.io/github
- https://docs.marimo.io/
