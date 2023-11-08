import os
import shutil
import yaml
import docker
import tarfile
import io
import sysconfig

# import copy

deploy_targets = [
    "docker",
    #'singularity',
    #'onefile'
]

mlcube_types = ["model", "metrics"]


def run_deployment(
    mlcubedir,
    outputdir,
    target,
    mlcube_type,
    entrypoint_script=None,
    configfile=None,
    modeldir=None,
    requires_gpu=None,
):
    """
    Run the deployment of the model.

    Args:
        mlcubedir (str): The path to the mlcube directory.
        outputdir (str): The path to the output directory.
        target (str): The target to deploy to.
        mlcube_type (str): Either 'model' or 'metrics'
        entrypoint_script (str): The path of entrypoint script. Only used for metrics and inference
        configfile (str, Optional): The path to the configuration file. Required for models
        modeldir (str, Optional): The path to the model directory. Required for models
        requires_gpu (str, Optional): Whether the model requires GPU. Required for models

    Returns:
        bool: True if the deployment was successful, False otherwise.
    """
    assert (
        target in deploy_targets
    ), f"The deployment target {target} is not a valid target."

    assert (
        mlcube_type in mlcube_types
    ), f"The mlcube type {mlcube_type} is not a valid type."

    if not os.path.exists(outputdir):
        os.makedirs(outputdir, exist_ok=True)

    assert not os.path.isfile(
        outputdir
    ), f"Output location {outputdir} exists but is a file, not a directory."

    if mlcube_type == "model":
        assert os.path.exists(
            modeldir
        ), f"Error: The model path {modeldir} does not exist."

        assert os.path.isdir(
            modeldir
        ), f"The model path {modeldir} exists but is not a directory."

        assert os.path.exists(
            configfile
        ), f"The config file {configfile} does not exist."

    if entrypoint_script:
        assert os.path.exists(
            entrypoint_script
        ), f"Error: The script path {entrypoint_script} does not exist."

    if target.lower() == "docker":
        result = deploy_docker_mlcube(
            mlcubedir, outputdir, entrypoint_script, configfile, modeldir, requires_gpu
        )
        assert result, "Something went wrong during platform-specific deployment."

    return True


def deploy_docker_mlcube(
    mlcubedir,
    outputdir,
    entrypoint_script=None,
    config=None,
    modeldir=None,
    requires_gpu=None,
):
    """
    Deploy the docker mlcube of the model or metrics calculator.

    Args:
        mlcubedir (str): The path to the mlcube directory.
        outputdir (str): The path to the output directory.
        entrypoint_script (str): The path of entrypoint script. Only used for metrics and inference
        config (str, Optional): The path to the configuration file. Required for models
        modeldir (str, Optional): The path to the model directory. Required for models
        requires_gpu (str, Optional): Whether the model requires GPU. Required for models
    """
    mlcube_config_file = os.path.join(mlcubedir, "mlcube.yaml")
    assert os.path.exists(mlcubedir) and os.path.exists(
        mlcube_config_file
    ), "MLCube Directory: This does not appear to be a valid MLCube directory."

    output_workspace_folder = os.path.join(outputdir, "workspace")
    mlcube_workspace_folder = os.path.join(mlcubedir, "workspace")

    os.makedirs(output_workspace_folder, exist_ok=True)
    if os.path.exists(mlcube_workspace_folder):
        shutil.copytree(
            mlcube_workspace_folder,
            output_workspace_folder,
            dirs_exist_ok=True,
        )

    if config is not None:
        shutil.copyfile(config, os.path.join(output_workspace_folder, "config.yml"))

    # First grab the existing the mlcube config
    if modeldir is not None:
        # we can use that as an indicator if we are doing model or metrics deployment
        mlcube_config = get_model_mlcube_config(
            mlcube_config_file, requires_gpu, entrypoint_script
        )
    else:
        mlcube_config = get_metrics_mlcube_config(mlcube_config_file, entrypoint_script)

    output_mlcube_config_path = os.path.join(outputdir, "mlcube.yaml")
    with open(output_mlcube_config_path, "w") as f:
        f.write(yaml.dump(mlcube_config, sort_keys=False, default_flow_style=False))

    # This tag will be modified later by our deployment mechanism
    docker_image = mlcube_config["docker"]["image"]

    # Run the mlcube_docker configuration process, forcing build from local repo
    gandlf_root = os.path.realpath(os.path.dirname(__file__) + "/../../")
    site_packages_dir = sysconfig.get_path("purelib")
    symlink_location = ""
    if (
        gandlf_root == site_packages_dir
    ):  # Installed via pip, not as editable source install, extra work is needed
        setup_files = ["setup.py", ".dockerignore", "pyproject.toml", "MANIFEST.in"]
        dockerfiles = [
            item
            for item in os.listdir(gandlf_root)
            if os.path.isfile(os.path.join(gandlf_root, item))
            and item.startswith("Dockerfile-")
        ]
        entrypoints = [
            item
            for item in os.listdir(gandlf_root)
            if os.path.isfile(os.path.join(gandlf_root, item))
            and item.startswith("gandlf_")
        ]
        for file in setup_files + dockerfiles + entrypoints:
            shutil.copy(
                os.path.join(gandlf_root, file),
                os.path.join(gandlf_root, "GANDLF", file),
            )
        if not os.path.exists(os.path.join(gandlf_root, "GANDLF", "GANDLF")):
            # point to same package directory, acts as a recursive location for the GaNDLF package
            symlink_location = os.path.join(gandlf_root, "GANDLF", "GANDLF")
            os.symlink("./", os.path.join(gandlf_root, "GANDLF", "GANDLF"))
        gandlf_root = os.path.join(gandlf_root, "GANDLF")

    print(os.listdir(gandlf_root))

    # Requires mlcube_docker python package to be installed with scripts available
    # command_to_run = "mlcube_docker configure --platform=docker  -Pdocker.build_strategy=always" + " --mlcube=" + os.path.realpath(mlcubedir) + " -Pdocker.build_context=" + gandlf_root
    command_to_run = (
        "mlcube_docker configure --platform=docker  -Prunner.build_strategy=always"
        + " --mlcube="
        + os.path.realpath(mlcubedir)
        + " -Prunner.build_context="
        + gandlf_root
    )

    print("Running MLCube configuration with the following command:")
    print(command_to_run)
    print(
        "If this is your first GaNDLF deployment, this may take longer than usual while image layers are built."
    )

    assert (
        os.system(command_to_run) == 0
    ), "mlcube_docker configuration failed. Check output for more details."

    # Container is made at this point, the recursive symlink no longer needs to exist
    if os.path.exists(symlink_location) and os.path.islink(symlink_location):
        os.unlink(symlink_location)

    # If mlcube_docker configuration worked, the image is now present in Docker so we can manipulate it.
    docker_client = docker.from_env()
    print("Connected to the docker service.")
    container = docker_client.containers.create(docker_image)

    # Embed mlcube config
    embed_asset(output_mlcube_config_path, container, "embedded_mlcube.yml")

    # Embed modeldir if available
    print("Attempting to embed the model...")
    embed_asset(modeldir, container, "embedded_model")

    # Embed config if available
    embed_asset(config, container, "embedded_config.yml")

    # Embed entrypoint script if available
    embed_asset(entrypoint_script, container, "entrypoint.py")

    # Commit the container to the same tag.
    docker_repo = docker_image.split(":")[0]
    docker_tag = docker_image.split(":")[1]
    container.commit(repository=docker_repo, tag=docker_tag)

    print(
        f"The updated container was committed successfully. It is available under Docker as: {docker_image} ."
    )
    print(
        f"This image should be distributed with the MLCube directory created at {outputdir}."
    )
    print(
        "You may now push this image (e.g. to Docker Hub) as normal. By doing so, the user will need only the MLCube directory in order to pull and run the image."
    )
    print(
        f"To run this container as an MLCube, you (or the end-user) should invoke the MLCube runner with --mlcube={outputdir} ."
    )
    print(
        "Otherwise, it will function as a standard docker image of GaNDLF, but with possibly embedded model/config files."
    )
    print("Deployment finished successfully!")
    return True


def get_metrics_mlcube_config(mlcube_config_file, entrypoint_script):
    """
    This function is used to get the metrics from mlcube config file.

    Args:
        mlcube_config_file (str): The path of mlcube config file.
        entrypoint_script (str): The path of entrypoint script.
    """
    mlcube_config = None
    with open(mlcube_config_file, "r") as f:
        mlcube_config = yaml.safe_load(f)
    if entrypoint_script:
        # modify the entrypoint to run a custom script
        mlcube_config["tasks"]["evaluate"]["entrypoint"] = "python3.9 /entrypoint.py"
    mlcube_config["docker"]["build_strategy"] = "auto"
    return mlcube_config


def get_model_mlcube_config(mlcube_config_file, requires_gpu, entrypoint_script):
    """
    This function returns the mlcube config for the model.

    Args:
        mlcube_config_file (str): Path to mlcube config file.
        requires_gpu (bool): Whether the model requires GPU.
        entrypoint_script (str): The path of entrypoint script. Only used for infer task
    """
    mlcube_config = None
    with open(mlcube_config_file, "r") as f:
        mlcube_config = yaml.safe_load(f)
    mlcube_config["docker"]["build_strategy"] = "auto"

    # modify for the embedded-model image.
    old_train_output_modeldir = mlcube_config["tasks"]["train"]["parameters"][
        "outputs"
    ].pop("modeldir", None)
    mlcube_config["tasks"]["infer"]["parameters"]["inputs"].pop("modeldir", None)
    mlcube_config["tasks"]["train"]["parameters"]["inputs"].pop("config", None)
    # Currently disabled because we've decided exposing config-on-inference complicates the MLCube use case.
    # mlcube_config["tasks"]["infer"]["parameters"]["inputs"].pop("config", None)

    # Change output so that each task always places secondary output in the workspace
    mlcube_config["tasks"]["train"]["parameters"]["outputs"][
        "output_path"
    ] = old_train_output_modeldir

    # Change entrypoints to point specifically to the embedded model and config
    mlcube_config["tasks"]["train"]["entrypoint"] = (
        mlcube_config["tasks"]["train"]["entrypoint"]
        + " --modeldir /embedded_model/"
        + " --config /embedded_config.yml"
    )
    mlcube_config["tasks"]["infer"]["entrypoint"] = (
        mlcube_config["tasks"]["infer"]["entrypoint"]
        + " --modeldir /embedded_model/"
        + " --config /embedded_config.yml"
    )

    # Change some configuration if GPU is required by default
    if requires_gpu:
        mlcube_config["platform"]["accelerator_count"] = 1
        mlcube_config["tasks"]["train"]["entrypoint"] = mlcube_config["tasks"]["train"][
            "entrypoint"
        ] = mlcube_config["tasks"]["train"]["entrypoint"].replace(
            "--device cpu", "--device cuda"
        )
        mlcube_config["tasks"]["infer"]["entrypoint"] = mlcube_config["tasks"]["infer"][
            "entrypoint"
        ] = mlcube_config["tasks"]["infer"]["entrypoint"].replace(
            "--device cpu", "--device cuda"
        )

    if entrypoint_script:
        # modify the infer entrypoint to run a custom script.
        device = "cuda" if requires_gpu else "cpu"
        mlcube_config["tasks"]["infer"][
            "entrypoint"
        ] = f"python3.9 /entrypoint.py --device {device}"

    return mlcube_config
    # Duplicate training task into one from reset (must be explicit) and one that resumes with new data
    # In either case, the embedded model will not change persistently.
    # The output in workspace will be the result of resuming training with new data on the embedded model.
    # This is currently disabled -- "reset" and "resume" seem to behave strangely in these conditions.
    # mlcube_config["tasks"]["training_from_reset"] = copy.deepcopy(
    #    mlcube_config["tasks"]["training"]
    # )
    # mlcube_config["tasks"]["training_from_reset"]["entrypoint"] = (
    #    mlcube_config["tasks"]["training_from_reset"]["entrypoint"] + " --reset True"
    # )
    # mlcube_config["tasks"]["training"]["entrypoint"] = (
    #    mlcube_config["tasks"]["training"]["entrypoint"] + " --resume True"
    # )


def embed_asset(asset, container, asset_name):
    """
    This function embeds an asset into a container.

    Args:
        asset (str): The path to the asset to embed.
        container (docker.models.containers.Container): The container to embed the asset into.
        asset_name (str): The name of the asset to embed.
    """
    # Tarball the modeldir, config.yml and mlcube.yaml into memory, insert into container
    if asset is not None:
        if os.path.exists(asset):
            asset_file_io = io.BytesIO()
            with tarfile.open(fileobj=asset_file_io, mode="w|gz") as tar:
                tar.add(os.path.realpath(asset), arcname=asset_name)
            asset_file_io.seek(0)
            container.put_archive("/", asset_file_io)
            asset_file_io.close()


## TODO!
## If implemented, even users who can't install singularity/docker/python could run frozen models.
## https://nuitka.net/ (Nuitka) can generate one-file binaries from python scripts, and can include additional files in the bundled self-extracting filesystem.
## https://pyinstaller.org/en/stable/ (PyInstaller) can do the same, albeit with a different approach.
## By bundling both GaNDLF and the required model/config files in one executable, we could have true drag-and-drop deployment of GaNDLF models.
## This is ideal, because then not even docker / singularity engines are needed.
## However, both of the above have occasional problems with certain packages (such as SimpleITK or Torch).
## In both cases, deployments need to be done on the corresponding platform/architecture, including separate CUDA/Torch versions.
## To implement this will need a lot of experimentation with each compiler/freezer and our own packages.
# def deploy_onefile(modeldir, config, outputdir):
#    raise NotImplementedError
