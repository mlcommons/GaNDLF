docker build -t gandlfcpu -f ../../../Dockerfile-CPU ../../..
mlcube configure --mlcube ./mlcube -Pdocker.build_strategy=always
