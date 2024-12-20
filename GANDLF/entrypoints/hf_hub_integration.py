import click
from GANDLF.entrypoints import append_copyright_to_help
from GANDLF.cli.huggingface_hub_handler import push_to_model_hub, download_from_hub
from pathlib import Path

huggingfaceDir_ = Path(__file__).parent.absolute()

huggingfaceDir = huggingfaceDir_.parent

# Huggingface template by default Path for the Model Deployment
huggingface_file_path = huggingfaceDir / "hugging_face.md"


@click.command()
@click.option(
    "--upload/--download",
    "-u/-d",
    required=True,
    help="Upload or download to/from a Huggingface Repo",
)
@click.option(
    "--repo-id",
    "-rid",
    required=True,
    help="Downloading/Uploading: A user or an organization name and a repo name separated by a /",
)
@click.option(
    "--token",
    "-tk",
    help="Downloading/Uploading: A token to be used for the download/upload",
)
@click.option(
    "--revision",
    "-rv",
    help="Downloading/Uploading: git revision id which can be a branch name, a tag, or a commit hash",
)
@click.option(
    "--cache-dir",
    "-cdir",
    help="Downloading: path to the folder where cached files are stored",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--local-dir",
    "-ldir",
    help="Downloading: if provided, the downloaded file will be placed under this directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--force-download",
    "-fd",
    is_flag=True,
    help="Downloading: Whether the file should be downloaded even if it already exists in the local cache",
)
@click.option(
    "--folder-path",
    "-fp",
    help="Uploading: Path to the folder to upload on the local file system",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--path-in-repo",
    "-pir",
    help="Uploading: Relative path of the directory in the repo. Will default to the root folder of the repository",
)
@click.option(
    "--commit-message",
    "-cr",
    help='Uploading: The summary / title / first line of the generated commit. Defaults to: f"Upload {path_in_repo} with huggingface_hub"',
)
@click.option(
    "--commit-description",
    "-cd",
    help="Uploading: The description of the generated commit",
)
@click.option(
    "--repo-type",
    "-rt",
    help='Uploading: Set to "dataset" or "space" if uploading to a dataset or space, "model" if uploading to a model. Default is model',
)
@click.option(
    "--allow-patterns",
    "-ap",
    help="Uploading: If provided, only files matching at least one pattern are uploaded.",
)
@click.option(
    "--ignore-patterns",
    "-ip",
    help="Uploading: If provided, files matching any of the patterns are not uploaded.",
)
@click.option(
    "--delete-patterns",
    "-dp",
    help="Uploading: If provided, remote files matching any of the patterns will be deleted from the repo while committing new files. This is useful if you don't know which files have already been uploaded.",
)
@click.option(
    "--hf-template",
    "-hft",
    help="Adding the template path for the model card: it is required during model upload",
    default=huggingface_file_path,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@append_copyright_to_help
def new_way(
    upload: bool,
    repo_id: str,
    token: str,
    hf_template: str,
    revision: str,
    cache_dir: str,
    local_dir: str,
    force_download: bool,
    folder_path: str,
    path_in_repo: str,
    commit_message: str,
    commit_description: str,
    repo_type: str,
    allow_patterns: str,
    ignore_patterns: str,
    delete_patterns: str,
):
    # """Manages model transfers to and from the Hugging Face Hub"""
    # """Manages model transfers to and from the Hugging Face Hub"""

    # # Ensure the hf_template is being passed and loaded correctly
    # template_path = Path(hf_template)

    # # Check if file exists and is readable
    # if not template_path.exists():
    #     raise FileNotFoundError(f"Model card template file '{hf_template}' not found.")

    # with template_path.open('r') as f:
    #     hf_template = f.read()

    # # Debug print the content to ensure it's being read
    # print(f"Template content: {type(hf_template)}...")  # Print the first 100 chars as a preview

    if upload:
        push_to_model_hub(
            repo_id,
            folder_path,
            hf_template,
            path_in_repo,
            commit_message,
            commit_description,
            token,
            repo_type,
            revision,
            allow_patterns,
            ignore_patterns,
            delete_patterns,
        )
    else:
        download_from_hub(
            repo_id, revision, cache_dir, local_dir, force_download, token
        )
