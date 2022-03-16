FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
LABEL github="https://github.com/CBICA/GaNDLF"
LABEL docs="https://cbica.github.io/GaNDLF/"
LABEL version=1.0
RUN apt-get update && apt-get install -y python3.7 python3-pip
COPY . /GaNDLF
WORKDIR /GaNDLF
RUN pip install -e .
# Entrypoint forces all commands given via "docker run" to go through python, CMD forces the default entrypoint script argument to be gandlf_run
# If a user calls "docker run gandlf:[tag] gandlf_anonymize", it will resolve to running "python gandlf_anonymize" instead.
# CMD is inherently overridden by args to "docker run", entrypoint is constant.
ENTRYPOINT python
CMD gandlf_run