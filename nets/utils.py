

def get_tf(module, obj_dict):
    """
    Retrieve an optimizer/loss object from the TF keras constructors
    via strings. Expects the following format for obj_dict:
        {"ObjectName": {"kwarg": kwargval, "kwarg2": kwargval}}

    For example:
        module = tf.keras.optimizers
        obj_dict = {"SGD": {"learning_rate": 0.001, "momentum": 0.01}

    :param module: TF module
    :param obj_dict: Object/kwarg dictionary as outlined above
    :return: TF object
    """
    if obj_dict is None:
        return None
    else:
        obj, kwargs = list(obj_dict.items())[0]
        return getattr(module, obj)(**kwargs)

