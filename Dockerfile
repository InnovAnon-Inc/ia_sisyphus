FROM ia_base as base
COPY ./ /tmp/py/
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install --no-cache-dir --upgrade ia_sisyphus
ENTRYPOINT ["python", "-m", "ia_sisyphus"]
