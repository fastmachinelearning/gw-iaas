import pytest

from hermes.quiver import Model, Platform


@pytest.mark.torch
def test_model(temp_repo, torch_model):
    from hermes.quiver.exporters import TorchOnnx

    model = Model("test", temp_repo, platform=Platform.ONNX)

    # make sure that a model entry got inserted
    # into the filesystem and that its attributes
    # remain empty
    assert model.name in temp_repo.fs.list()
    assert len(model.versions) == 0
    assert len(model.inputs) == 0
    assert len(model.outputs) == 0

    # make sure that the _find_exporter
    # method maps to the right exporter
    assert isinstance(model._find_exporter(torch_model), TorchOnnx)

    # export a version of the torch model
    export_path = model.export_version(
        torch_model,
        input_shapes={"x": [None, torch_model.size]},
        output_names=["y"],
    )
    assert export_path == temp_repo.fs.join(
        torch_model.name, "1", "model.onnx"
    )

    # make sure a version entry was created
    assert len(model.versions) == 1

    # make sure that the config was filled out
    # with all the right information
    assert model.config.input[0].name == "x"
    assert list(model.config.input[0].dims) == [-1, torch_model.size]

    # make sure the `inputs` attribute uses `None` syntax
    assert model.inputs["x"].shape == (None, torch_model.size)

    # if we now try to export a version and specify
    # the wrong size, this should raise an error
    with pytest.raises(ValueError):
        export_path = model.export_version(
            torch_model, input_shapes={"x": [None, torch_model.size + 1]}
        )

    # make sure that the new version directory
    # got deleted when the bad export failed
    assert len(model.versions) == 1

    # now verify that we can export a version
    # without specifying any auxiliary info
    export_path = model.export_version(torch_model)
    assert export_path == temp_repo.fs.join(model.name, "2", "model.onnx")

    # if we remove the 1th version of the model,
    # another call to `export_version` should still
    # increment to version `3`, after which point
    # we still have 2 versions associated with our model
    temp_repo.fs.remove(temp_repo.fs.join(model.name, "1"))
    export_path = model.export_version(torch_model)
    assert export_path == temp_repo.fs.join(model.name, "3", "model.onnx")
    assert len(model.versions) == 2


@pytest.mark.torch
def test_ensemble_model(temp_local_repo, torch_model):
    model1 = temp_local_repo.add("model-1", platform=Platform.ONNX)
    model1.export_version(
        torch_model, input_shapes={"x": [None, 10]}, output_names=["y"]
    )

    model2 = temp_local_repo.add("model-2", platform=Platform.ONNX)
    model2.export_version(
        torch_model, input_shapes={"x": [None, 10]}, output_names=["y"]
    )

    ensemble = Model("ensemble", temp_local_repo, platform=Platform.ENSEMBLE)
    ensemble.add_input(model1.inputs["x"])
    ensemble.pipe(model1.outputs["y"], model2.inputs["x"])
    ensemble.add_output(model2.outputs["y"])

    assert ensemble.config.input[0].name == "x"
    assert list(ensemble.config.input[0].dims) == [-1, 10]
    assert ensemble.inputs["x"].shape == (None, 10)

    assert ensemble.config.output[0].name == "y"
    assert list(ensemble.config.output[0].dims) == [-1, 10]
    assert ensemble.outputs["y"].shape == (None, 10)


@pytest.mark.torch
def test_ensemble_streaming(temp_local_repo, torch_model):
    model1 = temp_local_repo.add("model-1", platform=Platform.ONNX)
    model1.export_version(
        torch_model, input_shapes={"x": [None, 5, 10]}, output_names=["y"]
    )

    model2 = temp_local_repo.add("model-2", platform=Platform.ONNX)
    model2.export_version(
        torch_model, input_shapes={"x": [None, 4, 10]}, output_names=["y"]
    )

    ensemble = Model("ensemble", temp_local_repo, platform=Platform.ENSEMBLE)
    ensemble.add_streaming_inputs(
        [model1.inputs["x"], model2.inputs["x"]],
        stream_size=2,
    )

    assert ensemble.config.input[0].name == "stream"
    assert list(ensemble.config.input[0].dims) == [1, 9, 2]
