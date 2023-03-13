

def get_obj(module, obj_dict=None):
    """
    Instantiate an object from a given module with keyword args.
    Intended to aid JSON / string model configuration.

    Expects obj_dict to take the following form:
        {"ObjectName": {"kwarg": kwargval, "kwarg2": kwargval}}

    Example:
        get_tf(tf.keras.optimizers, {"SGD": {"learning_rate": 0.001,
            "momentum": 0.01})

    :param module: Module
    :param obj_dict: Dict
    :return: Instance of object
    """
    if obj_dict is None:
        return None
    else:
        obj, kwargs = list(obj_dict.items())[0]
        return getattr(module, obj)(**kwargs)
