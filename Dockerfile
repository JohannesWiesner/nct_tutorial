FROM continuumio/miniconda3:24.5.0-0

WORKDIR /app

COPY environment.yml requirements.txt ./
RUN conda env create -f environment.yml \
    && conda clean --all --yes

COPY app.py graph_utils.py nct_utils.py view_utils.py ./
COPY assets ./assets

ENV PATH=/opt/conda/envs/nct-tutorial/bin:$PATH \
    PYTHONUNBUFFERED=1

EXPOSE 8050

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--threads", "4", "app:server"]
