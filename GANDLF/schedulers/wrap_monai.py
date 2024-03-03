from monai.optimizers import WarmupCosineSchedule as WCS

def warmupcosineschedule(optimizers):
    # if not ("warmup_steps" in parameters["scheduler"]):
    #     parameters["scheduler"]["warmup_steps"] = 10
    
    return WCS(
        optimizers["model_optimizers"],
        t_total= 100,
        warmup_steps= 0.1*100
    )