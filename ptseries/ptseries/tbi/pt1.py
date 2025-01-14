from __future__ import annotations

import os
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import requests
from typing_extensions import TypeAlias, override

from ptseries.tbi.tbi_abstract import AsynchronousResultsBase, TBIDevice

if TYPE_CHECKING:
    FILE_LIKE: TypeAlias = str | os.PathLike


class PT1AsynchronousResults(AsynchronousResultsBase):
    """Initialize the AsynchronousResults object.

    Args:
        url: The URL of the PT-1 server.
        token: The authentication token.
        headers: The headers for the request.
        machine: The machine name.
        job_ids: The list or tuple of job IDs.
        n_samples: The number of samples.
        n_modes: The number of modes.
        n_tiling: The number of tiling. Defaults to 1.
        output_format: The output format. Defaults to "dict".
        save_dir: The directory to save the data. Defaults to None.
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
        super().__init__(
            url,
            token,
            headers,
            machine,
            job_ids,
            n_samples,
            n_modes,
            n_tiling,
            output_format,
            is_done,
            save_dir,
        )

    @override
    def cancel(self):
        """Cancel the asynchronous sampling."""
        cancelled_jobs = self.job_ids.copy()
        for job_id in self.job_ids:
            self.job_ids.remove(job_id)
        for job_id in cancelled_jobs:
            cancel_request_url = f"{self.url}/v1/cancel_job/{job_id}"
            _ = requests.get(
                url=cancel_request_url,
                json={"machine": self.machine},
                headers=self.headers,
                timeout=(3.05, 1),  # 3 seconds for connection, 10 seconds to cancel job
            )

    @override
    def get(self):
        """Get the results of the asynchronous sampling."""
        if self.job_ids:
            samples = np.empty((self.n_samples, self.n_tiling * self.n_modes), dtype=int)
            for i, job_id in enumerate(self.job_ids):
                samples[:, i * self.n_modes : (i + 1) * self.n_modes] = self._get_single_result(job_id)
            self.is_done = True
            return PT1.format_samples(samples, output_format=self.output_format)
        else:
            raise StopAsyncIteration("No jobs found.")

    def _get_single_result(self, job_id: str) -> npt.NDArray[np.int_]:
        """Get the results of a single job.

        Args:
            job_id: The job ID.
        """
        get_request_url = f"{self.url}/v1/get_job/{job_id}"

        n_trials = 0
        while n_trials < 5:
            r = requests.get(
                url=get_request_url,
                json={"machine": self.machine},
                headers=self.headers,
                timeout=(3.05, 60),  # 3 seconds for connection, 60 seconds for getting async samples
            )

            if r.status_code == 200:  # Request was successful
                result_json = r.json()

                em = result_json.get("error_message", None)
                js = result_json.get("job_status", None)

                if em is None and js is None:
                    samples = result_json["results"]
                    if self.save_dir is not None:
                        PT1._save_data(result_json, self.save_dir)
                    samples = PT1._reformat_samples(samples)
                    return samples
                elif js is not None:
                    raise RuntimeWarning(js)
                else:
                    raise StopAsyncIteration(em)

            elif r.status_code == 500:
                raise RuntimeError("An internal error occurred in the PT-1.")
            else:
                n_trials += 1

        raise ConnectionError("Could not connect to the hardware")


CONNECTION_UNSUCCESSFUL_MSG = "Connection unsuccessful"


class PT1(TBIDevice):
    """A class used to sample from real PT Series hardware.

    Args:
        n_loops: Number of loops in the TBI system. Defaults to 1.
        loop_lengths: Lengths of the loops in the TBI system. Defaults to None.
        postselected: Whether to use postselection. Defaults to True.
        ip_address: Deprecated: IP address of the PT-1 hardware, for example "0.0.0.0". Defaults to None.
        url: URL of the PT-1 hardware. Defaults to None.
        machine: Machine name of device to be used. Defaults to None.
        **kwargs: additional keyword arguments.

    Note:
        Deprecated in 2.6.0: `ip_address` will be removed in later versions, it is replaced by `url`.
    """

    descr = "PT-1"

    def __init__(
        self,
        n_loops: int = 1,
        loop_lengths: list[int] | tuple[int, ...] | npt.NDArray[np.int_] | None = None,
        postselected=True,
        ip_address: str | None = None,
        url: str | None = None,
        machine: str | None = None,
        **kwargs,
    ):
        super().__init__(
            n_loops=n_loops,
            loop_lengths=loop_lengths,
            postselected=postselected,
            ip_address=ip_address,
            url=url,
            machine=machine,
        )

        self.pt1_kwargs = kwargs

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        # Check if the device support asynchronous sampling
        sample_request_url = f"{self.url}/v1/submit"

        r = requests.post(
            url=sample_request_url,
            json={"machine": self.machine},
            headers=self.headers,
            timeout=(3.05, 3),  # 3 seconds for connection, 3 seconds to check if async is supported
        )

        if r.status_code == 404:
            self.sample_async_flag = False
        else:
            self.sample_async_flag = True

    @override
    def sample_async(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64],
        n_samples: int = 1,
        output_format: str = "dict",
        n_tiling: int = 1,
        save_dir: FILE_LIKE | None = None,
    ) -> PT1AsynchronousResults:
        r"""Asynchronously samples from the PT1 model.

        Args:
            input_state: Input state to be used. The left-most entry corresponds to the first mode entering the
                loop, for example :math:`(1,1,0)` for input state :math:`|110\rangle` Can be a list of integers, tuple
                of integers, or numpy array.
            theta_list: The list of beam splitter angles for sampling. Can be a list of floats, tuple of floats, or
                numpy array of float64.
            n_samples: The number of samples to draw. Defaults to 1.
            output_format: The format of the samples. Can be "dict", "tuple", "list" or "array". Defaults to "dict".
            n_tiling: The number of tilings to use. Defaults to 1.
            save_dir: Path to the directory in which to save results. If set to None the results are not saved. Defaults
                to None.

        Returns:
            AsynchronousResults: An object containing the asynchronous sampling results.

        Raises:
            ValueError: If the input state, theta list, or output format is invalid.
        """
        input_state, theta_list, output_format = self._validate_input(input_state, theta_list, output_format, n_tiling)

        # The samples are appended to raw_samples to be processed by tiling function
        job_ids = []
        theta_list_len = len(theta_list) // n_tiling
        temp_theta_list_len = theta_list_len

        # Sampler is run for each tiling
        for i in range(n_tiling):
            job_id = self._submit_job_async(
                input_state,
                theta_list[i * theta_list_len : temp_theta_list_len],
                n_samples,
            )

            job_ids.append(job_id)
            temp_theta_list_len += theta_list_len

        async_results = PT1AsynchronousResults(
            url=self.url,
            token=self.token,
            headers=self.headers,
            machine=self.machine,
            job_ids=job_ids,
            n_samples=n_samples,
            n_modes=len(input_state),
            save_dir=save_dir,
            n_tiling=n_tiling,
            output_format=output_format,
        )
        return async_results

    @override
    def sample(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64],
        n_samples: int = 1,
        output_format: str = "dict",
        n_tiling: int = 1,
        save_dir: FILE_LIKE | None = None,
    ) -> dict[tuple[int, ...], int] | tuple[tuple[int, ...], ...] | list[list[int]] | npt.NDArray[np.int_]:
        """Returns samples from the output of a real PT-1.

        Args:
            input_state: description of input modes. The left-most entry corresponds to the first mode entering the loop.
            theta_list: List of beam splitter angles
            n_samples: Number of samples to draw. Defaults to 1.
            output_format: Output format for the samples. "dict", "tuple", "list", "array". Defaults to "dict"
            n_tiling: Number of sample concatenation. Defaults to 1.
            save_dir: Path to the directory in which to save results. If set to None the results are not saved. Defaults
                to None.

        Returns:
            a dictionary of the form ``{state: counts}``
        """
        input_state, theta_list, output_format = self._validate_input(input_state, theta_list, output_format, n_tiling)
        if self.sample_async_flag:
            sl_time = 20e-3
            async_results = self.sample_async(
                input_state=input_state,
                theta_list=theta_list,
                n_samples=n_samples,
                output_format=output_format,
                n_tiling=n_tiling,
                save_dir=save_dir,
            )
            while not async_results.is_done:
                try:
                    time.sleep(sl_time)
                    samples = async_results.get()
                except RuntimeWarning:
                    sl_time = min(1, sl_time * 1.5)

                except KeyboardInterrupt as e:
                    for job_id in async_results.job_ids:
                        cancel_request_url = f"{self.url}/v1/cancel_job/{job_id}"
                        _ = requests.get(
                            url=cancel_request_url,
                            json={"machine": self.machine},
                            headers=self.headers,
                            timeout=(3.05, 10),  # 3 seconds for connection, 10 seconds to cancel job
                        )
                    raise KeyboardInterrupt("Sampling was interrupted by the user.")

            return samples
        else:
            # The samples are appended to raw_samples to be processed by tiling function
            samples = np.empty((n_samples, n_tiling * self.n_modes), dtype=int)
            theta_list_len = len(theta_list) // n_tiling
            temp_theta_list_len = theta_list_len

            # Sampler is run for each tiling
            for i in range(n_tiling):
                samples[:, i * self.n_modes : (i + 1) * self.n_modes] = self._request_samples(
                    input_state,
                    theta_list[i * theta_list_len : temp_theta_list_len],
                    n_samples,
                    save_dir=save_dir,
                )
                temp_theta_list_len += theta_list_len

            return self.format_samples(samples, output_format=output_format)

    def _validate_input(
        self,
        input_state: list[int] | tuple[int, ...] | npt.NDArray[np.int_],
        theta_list: list[float] | tuple[float, ...] | npt.NDArray[np.float64],
        output_format: str,
        n_tiling: int,
    ) -> tuple[list[int] | tuple[int, ...], list[float] | tuple[float, ...], str]:
        input_state = self._validate_input_state(input_state)
        theta_list = self._validate_theta_list(theta_list, n_tiling)

        if isinstance(self.loop_lengths, np.ndarray):
            self.loop_lengths = self.loop_lengths.tolist()

        if output_format not in ["tuple", "list", "array", "dict"]:
            output_format = "dict"
            warnings.warn('output_format must be "tuple", "list", "array" or "dict". Attempting with "dict"...')

        return input_state, theta_list, output_format

    def _validate_input_state(
        self, input_state_param: list[int] | tuple[int, ...] | npt.NDArray[np.int_]
    ) -> list[int] | tuple[int, ...]:
        if isinstance(input_state_param, np.ndarray):
            input_state = input_state_param.tolist()
        else:
            input_state = input_state_param

        if not all(0 <= value <= 1 for value in input_state):
            raise ValueError("All values in input modes must be between 0 and 1 (inclusive).")

        self.n_modes: int = len(input_state)
        if self.n_modes < max(self.loop_lengths) + 1:
            raise ValueError(
                f"The input_state must have at least as many modes as the longest {max(self.loop_lengths)  + 1}."
            )

        return input_state

    def _validate_theta_list(
        self, theta_list_param: list[float] | tuple[float, ...] | npt.NDArray[np.float64], n_tiling: int
    ) -> list[float] | tuple[float, ...]:
        if isinstance(theta_list_param, np.ndarray):
            theta_list = theta_list_param.tolist()
        else:
            theta_list = theta_list_param

        n_beam_splitters = self.calculate_n_beam_splitters(self.n_modes)

        if n_tiling < 1:
            raise ValueError("The minimal tiling number must be 1.")

        if len(theta_list) != n_beam_splitters * n_tiling:
            raise ValueError(
                f"List of thetas of length {n_beam_splitters * n_tiling} expected, received {len(theta_list)}."
            )

        return theta_list

    def test_connection(self):
        """Pings the PT-1 and prints whether this worked."""
        test_url = f"{self.url}/test-connection"

        try:
            r = requests.get(url=test_url, json={"machine": self.machine}, headers=self.headers, timeout=3)
            if r.status_code == 200:
                print("Connection successful")
            else:
                print(CONNECTION_UNSUCCESSFUL_MSG)
        except RuntimeError:
            print(CONNECTION_UNSUCCESSFUL_MSG)

    def get_hardware_status(self):
        """Gets the status of the PT-1 hardware and prints it."""
        status_url = f"{self.url}/status-single"

        try:
            r = requests.get(status_url, json={"machine": self.machine}, headers=self.headers, timeout=3)
        except RuntimeError:
            print(CONNECTION_UNSUCCESSFUL_MSG)
        else:
            if r.status_code == 200:
                result_json = r.json()
                if result_json["overall"]:
                    print("The hardware is ready")
                else:
                    print("The hardware is not ready yet.")
            else:
                print(CONNECTION_UNSUCCESSFUL_MSG)

    def get_device_certificate(self):
        """Gets the device certificate of the PT-1 hardware and uses it to validate SDK requests."""
        device_certificate_url = f"{self.url}/device-certificate"

        try:
            r = requests.get(
                url=device_certificate_url, json={"machine": self.machine}, headers=self.headers, timeout=3
            )
        except RuntimeError:
            raise RuntimeError(CONNECTION_UNSUCCESSFUL_MSG)
        else:
            if r.status_code == 200:
                result_json = r.json()
                self.device_certificate = result_json
            if r.status_code == 404:
                self.device_certificate = {
                    "max_modes": None,
                    "max_photons": None,
                    "max_samples": None,
                    "max_4photon_samples": None,
                    "loop_lengths": [1],
                    "version": None,
                }

    def _submit_job_async(
        self,
        input_state: list[int] | tuple[int, ...],
        bs_angles: list[float] | tuple[float, ...],
        n_samples: int,
    ) -> str:
        """Prepares and sends sample request to PT-1.

        Args:
            input_state: description of input modes. The left-most entry corresponds to the first mode entering the loop.
            bs_angles: list of beam splitter angles
            n_samples: number of samples to draw. Defaults to 1.
        """
        sample_request_url = f"{self.url}/v1/submit"

        json = {
            "input_state": input_state,
            "bs_angles": bs_angles,
            "n_samples": n_samples,
            "loop_lengths": self.loop_lengths,
            "postselected": self.postselected,
            "machine": self.machine,
            "extra_options": self.pt1_kwargs,
        }

        r = requests.post(
            url=sample_request_url,
            json=json,
            headers=self.headers,
            timeout=(3.05, 10),  # 3 seconds for connection, 10 seconds to get job id
        )
        if r.status_code == 200:  # Request was successful
            result_json = r.json()
            if result_json["job_id"] is not None:
                job_id = result_json["job_id"]
                return job_id
            else:
                raise RuntimeError("Could not get job id")
        elif r.status_code == 500:
            raise RuntimeError("An internal error occurred in the PT-1.")
        elif r.status_code == 404:
            raise requests.exceptions.HTTPError("The requested resource could not be found.")
        else:
            raise ConnectionError("Could not connect to the hardware")

    def _request_samples(
        self,
        input_state: list[int] | tuple[int, ...],
        bs_angles: list[float] | tuple[float, ...],
        n_samples: int,
        save_dir: FILE_LIKE | None = None,
    ) -> npt.NDArray[np.int_]:
        """Prepares and sends sample request to PT-1.

        Args:
            input_state: description of input modes.
                The left-most entry corresponds to the first mode entering the loop.
            bs_angles: list of beam splitter angles
            n_samples: number of samples to draw. Defaults to 1.
            save_dir: Path to the directory in which to save results. If set to None the results are not saved. Defaults
                to None.
        """
        sample_request_url = f"{self.url}/sample"

        json = {
            "input_state": input_state,
            "bs_angles": bs_angles,
            "n_samples": n_samples,
            "loop_lengths": self.loop_lengths,
            "postselected": self.postselected,
            "machine": self.machine,
            "extra_options": self.pt1_kwargs,
        }

        n_trials = 0
        while n_trials < 5:
            r = requests.post(
                url=sample_request_url,
                json=json,
                headers=self.headers,
                timeout=(3.05, 3600),  # 3 seconds for connection, 60 minutes for samples
            )

            if r.status_code == 200:  # Request was successful
                result_json = r.json()
                if result_json["error_message"] is None:
                    samples = result_json["results"]
                    if save_dir is not None:
                        self._save_data(result_json, save_dir)
                    samples = self._reformat_samples(samples)
                    return samples
                else:
                    raise RuntimeError(result_json["error_message"])
            elif r.status_code == 500:
                raise RuntimeError("An internal error occurred in the PT-1.")
            else:
                n_trials += 1
        raise ConnectionError("Could not connect to the hardware")
