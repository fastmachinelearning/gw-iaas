import argparse
import pathlib
import re


def main(project):
    project_dir = pathlib.PurePath(*project.split("."))
    project_dir = pathlib.PurePath("..", "projects") / project_dir

    with open(project_dir / "poetry.lock", "r") as f:
        lockfile = f.read()

    dockerfile = """
    FROM build
    COPY . /opt/build
    RUN set +x \\
            \\
            && mkdir /opt/lib \\
    """

    start = "\n" + " " * 8

    def add_lines(*lines):
        newlines = ""
        for line in lines:
            newlines += start + "\\" + start + f"&& {line} \\"
        return dockerfile + newlines

    regex = r"(../)+(libs/hermes/)(hermes\..+)"
    root = "/opt/gw-iaas/libs"
    for _, __, dep in re.findall(regex, lockfile):
        dockerfile = add_lines(
            f"cd {root}/{dep}", "poetry build", "cp dist/*.whl /opt/lib"
        )

    dockerfile = add_lines("cd /opt/build", "poetry build")
    dockerfile = dockerfile[:-2]

    print(dockerfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=str, help="Path to project")
    flags = parser.parse_args()
    main(**vars(flags))
