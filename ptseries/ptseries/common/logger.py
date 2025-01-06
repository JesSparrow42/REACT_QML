import os
from datetime import datetime
import json
import random
import string
import numpy as np
import warnings


class Logger:
    """Class that handles all logs for a training run

    When the logger is instantiated, a log folder is created to which all data is saved.
    By default, the log folder is created in the "data" folder, and if log_dir=None
    then no folder is created.

    Args:
        log_dir (str, optional): path to directory in which to create log folder. if set to None,
            no log folder is created
        log_tag (str, optional): tag added to log folder name
        metadata (dict, optional): dict of {str: value} where the user specified metadata
    """

    def __init__(self, log_dir="data", log_tag=None, metadata=None):
        self.log_dir = log_dir
        self.log_tag = log_tag
        self.metadata = metadata if metadata is not None else {}

        self.logs = {}
        self.log_keys = []

        if log_dir is not None:
            self._create_log_dir()
            self._save_metadata()

            # the git python package is optional, hence try/except
            try:
                self._register_git_commit()
            except:
                pass

    def log(self, key, iteration, value):
        """Logs a value for a given training/validation/testing iteration to a key

        Args:
            key (str): tag to identify the data
            iteration (int): iteration of the loop that yielded the data
            value (any numeric): value to log
        """
        if type(value) == np.ndarray:
            value = value.item()

        if key not in self.log_keys:
            self.log_keys.append(key)
            self.logs[key] = {}

        self.logs[key][iteration] = value

    def log_other(self, key, value):
        """Logs a single value identified by the key

        Args:
            key (str): tag to identify the data
            value (any numeric): value to log
        """
        self.logs[key] = value

    def register_metadata(self, metadata):
        """Adds metadata entries from a dict

        Args:
            metadata (dict): dict specifying the metadata entries to add
        """
        for key in metadata.keys():
            self.metadata[key] = metadata[key]
        if self.log_dir is not None:
            self._save_metadata()

    def save(self):
        """Saves log data to log folder"""
        if self.log_dir is not None:
            results_filename = self.log_folder + "/results.json"
            with open(results_filename, "w") as file:
                json.dump(self.logs, file)

        else:
            warnings.warn("The logger does not have a log_dir so was not saved to disk")

    def _create_log_dir(self):
        """Creates the log folder"""

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        now = datetime.now()
        current_time = now.strftime("%m-%d-%H-%M-%S")
        random_string = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        if self.log_tag is None:
            log_folder = self.log_dir + "/" + current_time + "_" + random_string
        else:
            log_folder = self.log_dir + "/" + self.log_tag + "_" + current_time + "_" + random_string

        self.log_folder = log_folder
        os.mkdir(log_folder)

    def _save_metadata(self):
        """Saves metadata as json to log folder"""

        metadata_filename = self.log_folder + "/metadata.json"

        with open(metadata_filename, "w") as file:
            json.dump(self.metadata, file)

    def _register_git_commit(self):
        """Adds current git commit reference to metadata"""

        # Note: git python package is optional
        import git

        repo = git.Repo(".")
        git_commit = repo.git.rev_parse("HEAD")
        self.register_metadata({"git_commit": git_commit})
