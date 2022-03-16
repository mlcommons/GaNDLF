FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
LABEL github="https://github.com/CBICA/GaNDLF"
LABEL docs="https://cbica.github.io/GaNDLF/"
LABEL version=1.0
RUN apt-get update && apt-get install -y python3.7 python3-pip
COPY . /GaNDLF
WORKDIR /GaNDLF
RUN pip install -e .
CMD python /GaNDLF/gandlf_verifyInstall