from monai.optimizers import WarmupCosineSchedule as WCS


def warmupcosineschedule(parameters):
    parameters["scheduler"]["warmup_steps"] = parameters["scheduler"].get(
        "warmup_steps", 0.1 * parameters["num_epochs"]
    )

    return WCS(
        parameters["optimizer_object"],
        t_total=parameters["num_epochs"],
        warmup_steps=parameters["scheduler"]["warmup_steps"],
    )
