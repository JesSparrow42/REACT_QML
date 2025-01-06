from __future__ import annotations

import json
import os
import random
import string
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    FILE_LIKE: TypeAlias = str | os.PathLike

import numpy as np
import numpy.typing as npt

from ptseries.tbi.representation.representation import Drawer


class AsynchronousResultsBase(ABC):
    """Abstract base class for retrieving asynchronous results from a specified device.

    Args:
        url: The URL of the device.
        token: The authentication token for accessing the device.
        headers: The headers for the request.
        machine: The machine name.
        job_ids: The IDs of the jobs to retrieve results from.
        n_samples: The number of samples.
        n_modes: The number of modes.
        n_tiling: The number of tiling. Defaults to 1.
        output_format: The output format. Defaults to "dict".
        is_done: Whether the asynchronous sampling jobs are done. Defaults to False.
        save_dir: The directory to save the results. Defaults to None.

    Raises:
        ValueError: If no job IDs are found.
    """

    def __init__(
        self,
        url: str,
        token: str | None,
        headers: dict[str, str] | None,
        machine: str | None,
        job_ids: list[str],
        n_samples: int,
        n_modes: int,
        n_tiling: int = 1,
        output_format: str = "dict",
        is_done: bool = False,
        save_dir: FILE_LIKE | None = None,
    ):
        self.url = url
        self.token = token
        self.headers = headers
        self.machine = machine
        self.job_ids = job_ids
        self.n_samples = n_samples
        self.n_modes = n_modes
        self.n_tiling = n_tiling
        self.output_format = output_format
        self.save_dir = save_dir
        self.is_done = is_done

        if self.job_ids is None:
            raise ValueError("No job ids found.")

    @abstractmethod
    def cancel(self):
        """Cancels the asynchronous sampling jobs."""
        pass

    @abstractmethod
    def get(self):
        """Retrieves the results from the specified device."""
        pass

    def ready(self):
        """Checks if the asynchronous sampling jobs are done."""
        return self.is_done

    def to_JSON(self, file: FILE_LIKE = "async_results.json") -> None:
        """Save the instance's attributes to a JSON string.

        Args:
            file: The path to the JSON file to save. It defaults to "async_results.json".

        Returns:
            None
        """
        json_dict = self.__dict__
        output_file = Path(file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w") as fp:
            json.dump(json_dict, fp, sort_keys=True, indent=4)

    def load_from_JSON(self, file: FILE_LIKE = "async_results.json") -> None:
        """Load the data from a JSON file and set the instance's attributes accordingly.

        Args:
            file: The path to the JSON file to load. It defaults to "async_results.json".

        Returns:
            None
        """
        with open(file, "r") as fp:
            json_dict = json.load(fp)
        for key, value in json_dict.items():
            setattr(self, key, value)

    @classmethod
    def create_from_JSON(cls, file: FILE_LIKE = "async_results.json") -> AsynchronousResultsBase:
        """Creates a new instance by loading the arguments from a JSON file.

        Args:
            file: The path to the JSON file that will be used to create the instance. It defaults to "async_results.json".

        Returns:
            An instance of the class initialized with the data from the JSON file.
        """
        with open(file, "r") as fp:
            json_dict = json.load(fp)
        return cls(**json_dict)


class TBIBase(ABC):
    """Base class for TBI (Time-Bin Interferometer) implementations.

    Args:
        n_loops: Number of loops in the TBI.
        loop_lengths: Lengths of each loop in the TBI. Defaults to None.
        postselected: Whether postselection is enabled. Defaults to False.
    """

    descr = ""

    def __init__(
        self,
        n_loops: int,
        loop_lengths: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | None = None,
        postselected: bool = False,
    ):
        self.n_loops = n_loops
        self.loop_lengths = loop_lengths or [1] * n_loops
        self.postselected = postselected

    @abstractmethod
    def sample(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64] | None,
        n_samples: int = 1,
        output_format: str = "dict",
        n_tiling: int = 1,
    ) -> dict[tuple[int, ...], int] | tuple[tuple[int, ...], ...] | list[list[int]] | npt.NDArray[np.int_]:
        r"""Returns samples from the output of a TBI.

        Args:
            input_state: input state to be used, for example :math:`(1,1,0)` for input state :math:`|110\rangle`
            theta_list: List of beam splitter angles.
            n_samples: Number of samples to draw. Defaults to 1.
            output_format: Output format for the samples. "dict", "tuple", "list", "array". Defaults to "dict"
            n_tiling: Number of sample concatenation. Defaults to 1.
        """
        pass

    def draw(
        self, input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_], padding: int = 1, show_plot: bool = True
    ):
        """Draws the representation of the TBI given the input state.

        Args:
            input_state: The input state to be represented in the TBI draw.
            padding: The padding size around the representation. Defaults to 1.
            show_plot: Whether to display the plot. Defaults to True.
        """
        n_modes = len(input_state)

        # N loops are sufficient to implement any N-mode unitary
        if self.descr == "fixed-random-unitary":
            self.n_loops = len(input_state) - 1
            self.loop_lengths = [1] * self.n_loops

        self.representation = Drawer()
        self.structure = self.representation.get_structure(n_modes, self.n_loops, loop_lengths=self.loop_lengths)
        if show_plot:
            self.representation.draw(self.structure, input_state, padding=padding)

    def calculate_n_beam_splitters(self, n_modes: int) -> int:
        """Calculates the number of beam splitters required for a given number of modes.

        Args:
            n_modes: The number of modes.

        Returns:
            int: The number of beam splitters required.
        """
        if self.loop_lengths is not None:
            n_beam_splitters = sum([n_modes - l for l in self.loop_lengths if l > 0])
        else:
            n_beam_splitters = n_modes - 1

        return n_beam_splitters

    @staticmethod
    def format_samples(
        samples: npt.NDArray[np.int_], output_format: str = "dict"
    ) -> dict[tuple[int, ...], int] | tuple[tuple[int, ...], ...] | list[list[int]] | npt.NDArray[np.int_]:
        """Formats the given samples based on the specified output format.

        Args:
            samples: The input samples to be formatted.
            output_format: The desired output format. Defaults to "dict".

        Returns:
            The formatted samples based on the specified output format.
        """
        if output_format == "tuple":
            return tuple(map(tuple, samples))

        elif output_format == "list":
            return list(map(list, samples))

        elif output_format == "array":
            return samples

        else:
            return {tuple(i.tolist()): int(j) for i, j in zip(*np.unique(samples, axis=0, return_counts=True))}

    @staticmethod
    def _save_data(result_json: dict, save_dir: FILE_LIKE):
        """Saves results to the specified directory.

        Args:
            result_json: the dict returned by the hardware
            save_dir: path to the directory where we save the dict as json
        """
        now = datetime.now()  # make the filename
        current_time = now.strftime("%m-%d-%H-%M-%S")
        random_string = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

        dir_exists = os.path.exists(save_dir)  # make directory if it doesn't exist
        if not dir_exists:
            os.mkdir(save_dir)

        filename = Path(save_dir) / f"{current_time}_{random_string}"  # save the file
        with open(filename, "w") as outfile:
            json.dump(result_json, outfile)


class TBISimulator(TBIBase):
    """A class representing a TBI (Time-Bin Interferometer) simulator.

    Args:
        n_loops: The number of loops in the TBI system.
        loop_lengths: The lengths of the loops in the TBI system. Defaults to None.
        postselected: Whether to use postselected sampling. Defaults to False.
        distinguishable: Whether the photons are distinguishable. Defaults to False.
        input_loss: The loss in the input state. Defaults to 0.0.
        bs_loss: The loss in the beam splitter. Defaults to 0.0.
        bs_noise: The noise in the beam splitter. Defaults to 0.0.
        detector_efficiency: The efficiency of the detectors. Defaults to 1.0.

    Raises:
        ValueError: If bs_loss is not between 0 and 1.
        ValueError: If input_loss is not between 0 and 1.
        ValueError: If detector_efficiency is not between 0 and 1.
        ValueError: If postselected is True and there is no finite probability of obtaining samples with the same number
            of photons as the input state.
    """

    descr = ""

    def __init__(
        self,
        n_loops: int,
        loop_lengths: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | None = None,
        postselected: bool = False,
        distinguishable: bool = False,
        input_loss: float = 0.0,
        bs_loss: float = 0.0,
        bs_noise: float = 0.0,
        detector_efficiency: float = 1.0,
    ):
        super().__init__(n_loops, loop_lengths, postselected)

        self.distinguishable = distinguishable
        self.input_loss = input_loss
        self.bs_loss = bs_loss
        self.bs_noise = bs_noise
        self.detector_efficiency = detector_efficiency

        if bs_loss < 0 or bs_loss > 1:
            raise ValueError("bs_loss must be a value between 0 and 1.")

        if input_loss < 0 or input_loss > 1:
            raise ValueError("input_loss must be a value between 0 and 1.")

        if detector_efficiency < 0 or detector_efficiency > 1:
            raise ValueError("detector_efficiency must be a value between 0 and 1.")

        if postselected:
            if np.isclose(detector_efficiency * (1 - bs_loss) * (1 - input_loss), 0):
                raise ValueError(
                    "If using postselected sampling, ensure that there is a finite probability of obtaining samples "
                    "with the same number of photons as input state e.g. ensure that input_loss or bs_loss are not "
                    "sets to 1."
                )


class TBIDevice(TBIBase):
    """A class representing a TBI (Time-Bin Interferometer) device.

    Args:
        n_loops: The number of loops in the TBI system.
        loop_lengths: The lengths of the loops in the TBI setup. Defaults to None.
        postselected: Whether to use postselected sampling. Defaults to False.
        ip_address: The IP address of the device. Defaults to None.
        url: The URL of the device. Defaults to None.
        machine: The machine name. Defaults to None.

    Raises:
        ValueError: If no URL is provided.
    """

    descr = ""

    def __init__(
        self,
        n_loops: int,
        loop_lengths: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | None = None,
        postselected: bool = False,
        ip_address: str | None = None,
        url: str | None = None,
        machine: str | None = None,
    ):
        super().__init__(n_loops, loop_lengths, postselected)

        self.machine = machine

        self.token: str | None = os.getenv("ORCA_AUTH_TOKEN")

        orca_url: str | None = os.getenv("ORCA_ACCESS_URL")

        if url is not None:
            self.url = url
        elif ip_address is not None:
            warnings.warn(
                "The 'ip_address' argument will be removed in future versions. Please use 'url' instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.url = f"http://{ip_address}:8080"
        elif orca_url:
            self.url = orca_url
        else:
            try:
                with open("ip_address.txt", "r") as file:
                    warnings.warn(
                        "The use of the 'ip_address.txt' file will be removed in future versions.\n"
                        "Please use the ORCA_ACCESS_URL environment variable instead.",
                        FutureWarning,
                        stacklevel=2,
                    )
                    contents = file.readlines()[0]
                    self.url = f"http://{contents.strip()}:8080"
                    return
            except FileNotFoundError:
                raise ValueError(
                    "A url must be provided.\nAlternatively set the environment variable 'ORCA_ACCESS_URL'"
                )

    @abstractmethod
    def sample_async(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64] | None,
        n_samples: int = 1,
        output_format: str = "dict",
        n_tiling: int = 1,
    ) -> AsynchronousResultsBase:
        r"""Returns asynchronous samples from the output of a TBI.

        Args:
            input_state: input state to be used, for example :math:`(1,1,0)` for input state :math:`|110\rangle`
            theta_list: List of beam splitter angles.
            n_samples: Number of samples to draw. Defaults to 1.
            output_format: Output format for the samples. "dict", "tuple", "list", "array". Defaults to "dict"
            n_tiling: Number of sample concatenation. Defaults to 1.
        """
        pass

    @staticmethod
    def _reformat_samples(samples: list[str]) -> npt.NDArray[np.int_]:
        """Converts json strings to numpy array.

        Args:
            samples: list of strings

        Returns:
            numpy array of integers
        """
        # Convert each string in the list to a list of integers
        reformatted_results = [[int(char) for char in item] for item in samples]
        # Convert the list of lists into a numpy array
        return np.array(reformatted_results)
