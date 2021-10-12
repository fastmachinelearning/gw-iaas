from multiprocessing import Queue
from typing import Optional, Sequence

from hermes.gwftools import FrameCrawler, FrameLoader, GCSFrameDownloader
from hermes.stillwater import InferenceClient
from hermes.typeo import typeo


class Preprocessor:
    def __init__(self, preproc_params):
        pass


@typeo("DeepClean client")
def main(
    data_dir: str,
    kernel_length: float,
    stride_length: float,
    sample_rate: float,
    inference_rate: float,
    channels: Sequence[str],
    sequence_id: int,
    url: str,
    model_name: str,
    model_version: int = 1,
    t0: Optional[float] = None,
    length: Optional[float] = None,
    preprocess_pkl: Optional[str] = None,
    timeout: Optional[float] = None,
) -> None:
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
        preprocessor = Preprocessor(preprocess_pkl)
    else:
        preprocessor = None

    # we want to be able to pass our strain data
    # directly to the postprocessing process, so
    # create a queue which our frame loader can
    # use to pass strain data and filenames
    strain_q = Queue()
    data_loader = FrameLoader(
        chunk_size=int(kernel_length * sample_rate),
        step_size=int(stride_length * sample_rate),
        sample_rate=sample_rate,
        channels=channels,
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
    client = InferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        profile=False,
        name="client",
    )

    # build a pipeline connecting all
    # the processes
    # TODO: this won't work since the __rshift__
    # returns a queue, need to figure out the
    # behavior that we want here. Dedicated Pipeline
    # object which starts all the processes?
    fname_source >> data_loader >> client
    with fname_source, data_loader, client:
        for output in client:
            continue
    return output


if __name__ == "__main__":
    main()
