FROM python:3.11
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin/
RUN pip install coverage hypothesis pytest poetry watchdog
ENV PYTHONPATH "${PYTHONPATH}:/app/src"
ENV PYTHONPYCACHEPREFIX "/tmp/pycache"
WORKDIR /app
COPY justfile requirements.txt .
RUN pip install -r requirements.txt
ENTRYPOINT ["just"]
# can be overriden for local development:
CMD ["run"]
