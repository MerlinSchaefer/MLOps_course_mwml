# MLOps_course_mwml

This repo contains my version of the [MadeWithML MLOps course](https://github.com/GokuMohandas/mlops-course).

I slightly modified some things but the steps are the same.
In my repo the explorative part is split into three directories with respective notebooks (`Design`, `Data`, `Modeling`), everything else can be found in the `app` directory.

## Structure within `app/`

```bash
tagifai/
├── data/         - training/testing data
├── venv/         - virtual environment
├── main.py       - training/optimization pipelines
└── utils.py      - supplementary utilities

```

## Virtual Environment

To use the package install [pyenv](https://github.com/pyenv/pyenv) and run the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
```
