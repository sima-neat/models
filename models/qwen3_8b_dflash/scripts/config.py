def get_layer_configuration(model_properties, layer):
    # Prequantized target weights retain their encoded precision.
    return {"precision": "A_BF16_W_INT8" if model_properties["is_draft_model"] else "BF16"}
