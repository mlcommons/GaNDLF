import os
import shutil
import yaml
import docker 
import tarfile
import io

deploy_targets = [
    'docker',
    #'singularity',
    #'onefile'
    ]

def run_deployment(modeldir, configfile, target, outputdir, mlcubedir):
    if target not in deploy_targets:
        print(f"Error: The deployment target {target} is not a valid target.")
        return False
        
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir, exist_ok=True)
        
    if os.path.isfile(args.outputdir):
        print(f"Error: Output location {args.outputdir} exists but is a file, not a directory.")
        return False
    if not os.path.exists(modeldir):
        print(f"Error: The model path {modeldir} does not exist.")
        return False
    if not os.path.isdir(modeldir):
        print(f"Error: The model path {modeldir} exists but is not a directory.")
        return False
    if not os.path.exists(configfile):
        print(f"Error: the config file {configfile} does not exist.")
        return False
    
    if target.tolower() == 'docker':
        if not deploy_docker_mlcube(modeldir, config, outputdir, mlcubedir):
            print("Error: Something went wrong during platform-specific deployment.")
            return False
    
    return True
    
def deploy_docker_mlcube(modeldir, config, outputdir, mlcubedir):
    # Set up docker client for any future calls
    docker_client = docker.from_env()
    
    mlcube_config_file = mlcubedir+"/mlcube.yaml"
    if not os.path.exists(mlcube_config_file):
        print("Error: This does not appear to be a valid MLCube directory.")
        return False
    
    os.makedirs(args.outputdir+"/workspace", exist_ok=True)
    shutil.copytree(mlcubedir+"/workspace", outputdir+"/workspace", dirs_exist_ok=True)
    shutil.copyfile(config, outputdir+"/workspace/config.yml")
    
    
    
    # First grab the existing the mlcube config then modify for the embedded-model image.
    mlcube_config = None
    with open(mlcube_config_file, "r") as f:
        mlcube_config = yaml.safe_load(f)
    
    output_mlcube_config_path = outputdir+"/mlcube.yaml"
    
    del mlcube_config["tasks"]["training"]["parameters"]["outputs"]["modelDir"]
    del mlcube_config["tasks"]["inference"]["parameters"]["inputs"]["modelDir"]
    #del mlcube_config["tasks"]["training"]["parameters"]["inputs"]["config"]
    #del mlcube_config["tasks"]["inference"]["parameters"]["inputs"]["config"]
    
    mlcube_config["tasks"]["training"]["parameters"]["outputs"]["outputdir"] = {"type": "directory", "default":"/model"}
    mlcube_config["tasks"]["inference"]["parameters"]["outputs"]["outputdir"] = {"type": "directory", "default":"/inference"}
    
    mlcube_config["tasks"]["training"]["entrypoint"] = mlcube_config["tasks"]["training"]["entrypoint"] + " --modelDir /embedded_model/"
    
    mlcube_config["tasks"]["inference"]["entrypoint"] = mlcube_config["tasks"]["inference"]["entrypoint"] + " --modelDir /embedded_model/"
    
    mlcube_config["docker"]["build_strategy"] = "auto"
    
    with open(output_mlcube_config_path, "w" as f:
        f.write(yaml.dump(mlcube_config, default_flow_style=False))
    
    # This tag will be modified later by our deployment mechanism
    docker_image = mlcube_config["docker"]["image"]
    
    # Run the mlcube_docker configuration process, forcing build from local repo
    gandlf_root = os.path.realpath(os.path.dirname(__file__) + "/../../")
    # Requires mlcube_docker python package to be installed with scripts available
    command_to_run = "mlcube_docker configure --platform=docker  -Pdocker.build_strategy=always"
        + " --mlcube=" + os.path.realpath(mlcubedir)
        + " -Pdocker.build_context=" + gandlf_root
    
    if os.system(command_to_run) > 0:
        print("Error: mlcube_docker configuration failed. Check output for more details.")
        return False
        
    # If mlcube_docker configuration worked, the image is now present in Docker so we can manipulate it.
    container = docker_client.containers.create(docker_tag)
    
    
    # Tarball the modeldir, config.yml and mlcube.yaml into memory, insert into container
    modeldir_file_io = io.BytesIO()
    with tarfile.open(fileobj=modeldir_file_io, mode="w|gz") as tar:
        tar.add(os.path.realpath(modeldir), arcname="embedded_model")
    modeldir_file_io.seek(0)
    container.put_archive("/", modeldir_file_io)
    modeldir_file_io.close()
    
    # Do the same for config and mlcube.yaml
    config_file_io = io.BytesIO()
    with tarfile.open(fileobj=config_file_io, mode="w|gz") as tar:
        tar.add(os.path.realpath(config), arcname="embedded_config.yml")
    config_file_io.seek(0)
    container.put_archive("/", config_file_io)
    config_file_io.close()
    
    mlcube_file_io = io.BytesIO()
    with tarfile.open(fileobj=mlcube_file_io, mode="w|gz") as tar:
        tar.add(os.path.realpath(output_mlcube_config_path), arcname="embedded_mlcube.yaml")
    mlcube_file_io.seek(0)
    container.put_archive("/", mlcube_file_io)
    mlcube_file_io.close()
    
    # Commit the container to the same tag.
    docker_repo = docker_image.split(":")[0]
    docker_tag = docker_image.split(":")[1]
    container.commit(repository=docker_repo, tag=docker_tag)
    
    print(f"The updated container was committed successfully. It is available under Docker as: {docker_image} .")
    
    return True
    
    
    
    
## TODO!
## If implemented, even users who can't install singularity/docker/python could run frozen models.
def deploy_onefile(modeldir, config, outputdir):
    raise NotImplementedError