from huggingface_hub import HfApi, snapshot_download
from typing import List, Union


def push_to_model_hub(
    repo_id: str,
    folder_path: str,
    path_in_repo: Union[str, None] = None,
    commit_message: Union[str, None] = None,
    commit_description: Union[str, None] = None,
    token: Union[str, None] = None,
    repo_type: Union[str, None] = None,
    revision: Union[str, None] = None,
    allow_patterns: Union[List[str], str, None] = None,
    ignore_patterns: Union[List[str], str, None] = None,
    delete_patterns: Union[List[str], str, None] = None,
):
    api = HfApi(token=token)

    api.create_repo(repo_id, exist_ok=True)

    api.upload_folder(
        repo_id=repo_id,
        token=token,
        folder_path=folder_path,
        path_in_repo=path_in_repo,
        commit_message=commit_message,
        commit_description=commit_description,
        repo_type=repo_type,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        delete_patterns=delete_patterns,
    )


def download_from_hub(
    repo_id: str,
    revision: Union[str, None] = None,
    cache_dir: Union[str, None] = None,
    local_dir: Union[str, None] = None,
    force_download: bool = False,
    token: Union[str, None] = None,
):
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        force_download=force_download,
        token=token,
    )
