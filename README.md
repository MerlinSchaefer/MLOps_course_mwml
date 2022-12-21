# MLOps_course_mwml

This repo contains my version of the [MadeWithML MLOps course](https://github.com/GokuMohandas/mlops-course).

I slightly modified some things but the steps are the same.
In my repo the explorative part is split into three directories with respective notebooks (`Design`, `Data`, `Modeling`), everything else can be found in the `app` directory.

## Structure within `app/`

```bash

├── venv/            - virtual environment (needs to be created first)
├── config/          - config settings
├── docs/            - interative documentation (see below)
├── tagifai/
    ├── data.py      - data processing utilities
    ├── evaluate.py  - evaluation components
    ├── main.py      - training/optimization operations
    ├── predict.py   - inference utilities
    └── train.py     - training utilities
├── setup.py         - setup script
├── requirements.txt - training/optimization pipelines
├── setup.py         - setup script
└── utils.py         - supplementary utilities
```

## Virtual Environment

To use the package install [pyenv](https://github.com/pyenv/pyenv) and run the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
```

## Documentation

To view the documentation navigate to the `app/` directory activate the virtual environment and run:

```bash
python3 -m mkdocs serve
```
