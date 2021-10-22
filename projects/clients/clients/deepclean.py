import logging
from multiprocessing import Queue
from typing import Optional, Sequence, Union

from clients.utils import FrameWriter, Preprocessor, get_logger

from hermes.gwftools import FrameCrawler, FrameLoader, GCSFrameDownloader
from hermes.stillwater import InferenceClient
from hermes.typeo import typeo


@typeo("DeepClean client")
def main(
    data_dir: str,
    write_dir: str,
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    inference_rate: float,
    channels: Union[str, Sequence[str]],
    sequence_id: int,
    url: str,
    model_name: str,
    model_version: int = 1,
    t0: Optional[float] = None,
    length: Optional[float] = None,
    preprocess_pkl: Optional[str] = None,
    timeout: Optional[float] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Clean a stretch of data using an inference service"""

    # configure logging up front
    _ = get_logger(log_file, verbose)
    logger = logging.getLogger()

    if isinstance(channels, str) or len(channels) == 1:
        if not isinstance(channels, str):
            channels = channels[0]
        with open(channels, "r") as f:
            channels = [i for i in f.read().splitlines() if i]

    # source for frame filenames will be different
    # depending on whether our data is local or in
    # the cloud
    if data_dir.startswith("gs://"):
        # for cloud data, download to local files and
        # pass the names of these downloaded files to
        # frame loading process
        root = data_dir.replace("gs://", "")
        fname_source = GCSFrameDownloader(
            root=root, t0=t0, length=length, name="fname-source"
        )

        # we don't need these frames locally once they're
        # loaded, so leave that to the frame loader
        remove = True
    else:
        # for local replay data, create a process which
        # monitors a local directory for new frames and
        # passes the names of those frames to the frame loader
        fname_source = FrameCrawler(
            data_dir, timeout=timeout, name="fname-source"
        )

        # these are likely being managed by a process
        # that isn't ours to interfere with, so don't
        # delete the files onces we're done with them
        remove = False

    # if we specified a DeepClean preprocessing
    # pickle, build a callable object which can
    # perform the requisite preprocessing
    if preprocess_pkl is not None:
        preprocessor = Preprocessor(preprocess_pkl, sample_rate)
    else:
        preprocessor = None

    # we want to be able to pass our strain data
    # directly to the postprocessing process, so
    # create a queue which our frame loader can
    # use to pass strain data and filenames
    strain_q = Queue()
    data_loader = FrameLoader(
        chunk_size=int(stride_length * sample_rate),
        step_size=int(stride_length * sample_rate),
        sample_rate=sample_rate,
        channels=channels,
        t0=t0,
        length=length,
        sequence_id=sequence_id,
        preprocessor=preprocessor,
        remove=remove,
        strain_q=strain_q,
        rate=inference_rate,
        name="frame-loader",
    )

    # now create a client process which will take
    # the streams output by the data loader
    # and package them up for the server
    # TODO: include an ignore_streams arg
    # to the inference client in case we want
    # to make requests to the streaming ensemble
    # in an arbitrary order
    logger.info("creating client")
    client = InferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        profile=False,
        name="client",
    )
    logger.info("client created")

    writer = FrameWriter(
        write_dir=write_dir,
        channel_name=channels[0],
        step_size=int(stride_length * sample_rate),
        sample_rate=sample_rate,
        strain_q=strain_q,
        preprocessor=preprocessor,
        name="writer",
    )

    # build a pipeline connecting all the processes
    with fname_source >> data_loader >> client >> writer as pipeline:
        for fname in pipeline:
            logger.info(f"Processed frame {fname}")


if __name__ == "__main__":
    main()
