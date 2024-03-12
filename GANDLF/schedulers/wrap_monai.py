from monai.optimizers import WarmupCosineSchedule as WCS

def warmupcosineschedule(parameters):
    return WCS(
        parameters["optimizer_object"],
        t_total = parameters["num_epochs"],
        warmup_steps = 0.1 * parameters["num_epochs"],
    )
