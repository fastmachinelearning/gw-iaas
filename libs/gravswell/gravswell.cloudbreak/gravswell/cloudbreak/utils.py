import logging
import queue
import threading
import time
import typing


def wait_for(
    callback: typing.Callable,
    msg: typing.Optional[str] = None,
    exit_msg: typing.Optional[str] = None,
):
    q = queue.Queue()

    def target(msg):
        try:
            while True:
                release = callback()
                try:
                    msg, release = release
                except (TypeError, ValueError):
                    pass

                q.put((msg, release))
                if release:
                    break
                time.sleep(0.5)
        except Exception as e:
            q.put((e, True))

    t = threading.Thread(target=target, args=(msg,))
    t.start()

    i, line_length = 0, 0
    level = logging.INFO
    try:
        while True:
            try:
                msg, release = q.get_nowait()
                if isinstance(msg, Exception):
                    raise msg
                elif release:
                    break
            except queue.Empty:
                continue

            if msg is None:
                continue

            dots = "." * (i + 1)
            line_length = max(len(msg) + len(dots), line_length)
            spaces = " " * (line_length - len(msg) - len(dots))

            print(msg + dots + spaces, end="\r", flush=True)
            time.sleep(0.5)
            i = (i + 1) % 3

    except Exception:
        exit_msg = "Encountered error in callback"
        level = logging.ERROR
        raise
    finally:
        # clear out the existing msg
        print(" " * line_length, end="\r", flush=True)
        if exit_msg is not None:
            logging.log(level, exit_msg)

    return release
