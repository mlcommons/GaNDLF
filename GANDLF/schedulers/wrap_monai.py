from monai.optimizers import WarmupCosineSchedule as WCS

def warmupcosineschedule(parameters):
    return WCS(
        parameters["optimizer_object"], #optimizers["model_optimizers"],
        t_total= 100,
        warmup_steps= 0.1*100
    )