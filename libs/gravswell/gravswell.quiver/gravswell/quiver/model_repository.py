import re
from itertools import count

from gravswell.quiver import Model, io, platforms


class ModelRepository:
    def __init__(self, root: str, **kwargs) -> None:
        if root.startswith("gs://"):
            self.fs = io.GCSFileSystem(root, **kwargs)
        else:
            self.fs = io.LocalFileSystem(root)

        self.models = {}
        for model in self.list(""):
            try:
                config = self.read_config(self.join(model, "config.pbtxt"))
            except FileNotFoundError:
                raise ValueError(
                    "Failed to initialize repo at {} due to "
                    "model with missing config {}".format(self.root, model)
                )

            try:
                platform = platforms.platforms[config.platform]
            except KeyError:
                raise ValueError(
                    "Failed to initialize repo at {} due to "
                    "model {} with unknown platform {}".format(
                        self.root, model, config.platform
                    )
                )
            self.add(model, platform)

    @property
    def models(self) -> dict:
        return {model.name: model for model in self._models}

    def add(
        self, name: str, platform: platforms.Platform, force: bool = False
    ) -> Model:
        if name in self.models and not force:
            raise ValueError("Model {} already exists".format(name))
        elif name in self.models:
            # append an index to the name of the model starting at 0
            pattern = re.compile(f"{name}_[0-9]+")
            matches = list(filter(pattern.fullmatch, self.models))

            if len(matches) == 0:
                # no postfixed models have been made yet, start at 0
                index = 0
            else:
                # search for the first available index
                pattern = re.compile(f"(?<={name}_)[0-9]+")
                postfixes = [int(pattern.search(x).group(0)) for x in matches]
                for index, postfix in zip(count(0), sorted(postfixes)):
                    if index != postfix:
                        break
                else:
                    # indices up to len(matches) are taken,
                    # increment to the next available
                    index += 1
            name += f"_{index}"

        model = Model(name=name, repository=self, platform=platform)
        self._models.append(model)
        return model

    def remove(self, model_name: str):
        try:
            model = self.models.pop(model_name)
        except KeyError:
            raise ValueError(f"Unrecognized model {model_name}")
        self.fs.remove(model.name)

    def delete(self):
        model_names = self.models.keys()
        for model_name in model_names:
            self.remove(model_name)
        self.fs.delete()
