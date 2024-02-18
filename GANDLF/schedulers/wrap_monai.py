from monai.optimizers import WarmupCosineSchedule as WCS

def warmupcosineschedule(optimizers):
    return WCS(
        optimizers["model_optimizers"],
        
    )