from multiprocessing import Queue
from typing import Optional

from dc_online_prod.utils import (
    FrameWriter,
    Preprocessor,
    TwoFileFrameCrawler,
    TwoFileFrameLoader,
    get_logger,
)

from hermes.stillwater import InferenceClient
from hermes.typeo import typeo


@typeo("DeepClean online client")
def main(
    witness_data_dir: str,
    strain_data_dir: str,
    write_dir: str,
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    inference_rate: float,
    channels: str,
    preprocess_pkl: str,
    sequence_id: int,
    url: str,
    model_name: str,
    model_version: int = 1,
    max_latency: Optional[float] = None,
    num_frames: Optional[int] = None,
    start_first: bool = False,
    timeout: Optional[float] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Clean a stretch of data using an inference service"""

    # configure logging up front
    logger = get_logger(log_file, verbose)

    # read in channels from file
    with open(channels, "r") as f:
        channels = [i for i in f.read().splitlines() if i]

    # for local replay data, create a process which
    # monitors a local directory for new frames and
    # passes the names of those frames to the frame loader
    fname_source = TwoFileFrameCrawler(
        witness_data_dir,
        strain_data_dir,
        timeout=timeout,
        N=num_frames,
        start_first=start_first,
        name="fname-source",
    )

    # build a callable object which can
    # perform the requisite preprocessing
    preprocessor = Preprocessor(preprocess_pkl, sample_rate)

    # we want to be able to pass our strain data
    # directly to the postprocessing process, so
    # create a queue which our frame loader can
    # use to pass strain data and filenames
    strain_q = Queue()
    data_loader = TwoFileFrameLoader(
        chunk_size=int(stride_length * sample_rate),
        step_size=int(stride_length * sample_rate),
        sample_rate=sample_rate,
        channels=channels,
        sequence_id=sequence_id,
        preprocessor=preprocessor,
        remove=False,
        strain_q=strain_q,
        rate=inference_rate,
        name="frame-loader",
    )

    # now create a client process which will take
    # the streams output by the data loader
    # and package them up for the server
    client = InferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        profile=False,
        name="client",
    )

    if max_latency is not None:
        throw_away = max_latency // stride_length
    else:
        throw_away = None

    writer = FrameWriter(
        write_dir=write_dir,
        channel_name=channels[0],
        step_size=int(stride_length * sample_rate),
        sample_rate=sample_rate,
        strain_q=strain_q,
        throw_away=throw_away,
        preprocessor=preprocessor,
        name="writer",
    )

    # build a pipeline connecting all the processes
    pipeline = fname_source >> data_loader >> client >> writer
    with pipeline:
        for fname, latency in pipeline:
            logger.info(f"Processed frame {fname} with latency {latency:0.4f}")


if __name__ == "__main__":
    main()
