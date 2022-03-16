FROM pytorch/manylinux-cuda102
LABEL github="https://github.com/CBICA/GaNDLF"
LABEL docs="https://cbica.github.io/GaNDLF/"
LABEL version=1.0
COPY . /GaNDLF
WORKDIR /GaNDLF
RUN pip install -e .
CMD python /GaNDLF/gandlf_verifyInstall