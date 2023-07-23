
��root"_tf_keras_network*�{"name": "model_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sla"}, "name": "sla", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "product_type"}, "name": "product_type", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "brand"}, "name": "brand", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "service_type"}, "name": "service_type", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "incident_type"}, "name": "incident_type", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "open_to_close_hour"}, "name": "open_to_close_hour", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "response_to_resolved_hour"}, "name": "response_to_resolved_hour", "inbound_nodes": []}, {"class_name": "StringLookup", "config": {"name": "string_lookup_60", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8"}, "name": "string_lookup_60", "inbound_nodes": [[["sla", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_61", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8"}, "name": "string_lookup_61", "inbound_nodes": [[["product_type", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_62", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8"}, "name": "string_lookup_62", "inbound_nodes": [[["brand", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_63", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8"}, "name": "string_lookup_63", "inbound_nodes": [[["service_type", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_64", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8"}, "name": "string_lookup_64", "inbound_nodes": [[["incident_type", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_24", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "name": "normalization_24", "inbound_nodes": [[["open_to_close_hour", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_25", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "name": "normalization_25", "inbound_nodes": [[["response_to_resolved_hour", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_60", "trainable": true, "dtype": "float32", "num_tokens": 8, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_60", "inbound_nodes": [[["string_lookup_60", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_61", "trainable": true, "dtype": "float32", "num_tokens": 11, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_61", "inbound_nodes": [[["string_lookup_61", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_62", "trainable": true, "dtype": "float32", "num_tokens": 24, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_62", "inbound_nodes": [[["string_lookup_62", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_63", "trainable": true, "dtype": "float32", "num_tokens": 3, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_63", "inbound_nodes": [[["string_lookup_63", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_64", "trainable": true, "dtype": "float32", "num_tokens": 22, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_64", "inbound_nodes": [[["string_lookup_64", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_10", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_10", "inbound_nodes": [[["normalization_24", 0, 0, {}], ["normalization_25", 0, 0, {}], ["category_encoding_60", 0, 0, {}], ["category_encoding_61", 0, 0, {}], ["category_encoding_62", 0, 0, {}], ["category_encoding_63", 0, 0, {}], ["category_encoding_64", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 70, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["concatenate_10", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["dense_20", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["dropout_10", 0, 0, {}]]]}], "input_layers": [["open_to_close_hour", 0, 0], ["response_to_resolved_hour", 0, 0], ["sla", 0, 0], ["product_type", 0, 0], ["brand", 0, 0], ["service_type", 0, 0], ["incident_type", 0, 0]], "output_layers": [["dense_21", 0, 0]]}, "shared_object_id": 27, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "open_to_close_hour"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "response_to_resolved_hour"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "sla"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "product_type"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "brand"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "service_type"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "incident_type"]}]], {}]}, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "open_to_close_hour"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "response_to_resolved_hour"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "sla"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "product_type"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "brand"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "service_type"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "incident_type"]}], "keras_version": "2.11.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sla"}, "name": "sla", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "product_type"}, "name": "product_type", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "brand"}, "name": "brand", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "service_type"}, "name": "service_type", "inbound_nodes": [], "shared_object_id": 3}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "incident_type"}, "name": "incident_type", "inbound_nodes": [], "shared_object_id": 4}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "open_to_close_hour"}, "name": "open_to_close_hour", "inbound_nodes": [], "shared_object_id": 5}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "response_to_resolved_hour"}, "name": "response_to_resolved_hour", "inbound_nodes": [], "shared_object_id": 6}, {"class_name": "StringLookup", "config": {"name": "string_lookup_60", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "name": "string_lookup_60", "inbound_nodes": [[["sla", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "StringLookup", "config": {"name": "string_lookup_61", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "name": "string_lookup_61", "inbound_nodes": [[["product_type", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "StringLookup", "config": {"name": "string_lookup_62", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "name": "string_lookup_62", "inbound_nodes": [[["brand", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "StringLookup", "config": {"name": "string_lookup_63", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "name": "string_lookup_63", "inbound_nodes": [[["service_type", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "StringLookup", "config": {"name": "string_lookup_64", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "name": "string_lookup_64", "inbound_nodes": [[["incident_type", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Normalization", "config": {"name": "normalization_24", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "name": "normalization_24", "inbound_nodes": [[["open_to_close_hour", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Normalization", "config": {"name": "normalization_25", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "name": "normalization_25", "inbound_nodes": [[["response_to_resolved_hour", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_60", "trainable": true, "dtype": "float32", "num_tokens": 8, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_60", "inbound_nodes": [[["string_lookup_60", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_61", "trainable": true, "dtype": "float32", "num_tokens": 11, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_61", "inbound_nodes": [[["string_lookup_61", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_62", "trainable": true, "dtype": "float32", "num_tokens": 24, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_62", "inbound_nodes": [[["string_lookup_62", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_63", "trainable": true, "dtype": "float32", "num_tokens": 3, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_63", "inbound_nodes": [[["string_lookup_63", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_64", "trainable": true, "dtype": "float32", "num_tokens": 22, "output_mode": "multi_hot", "sparse": false}, "name": "category_encoding_64", "inbound_nodes": [[["string_lookup_64", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Concatenate", "config": {"name": "concatenate_10", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_10", "inbound_nodes": [[["normalization_24", 0, 0, {}], ["normalization_25", 0, 0, {}], ["category_encoding_60", 0, 0, {}], ["category_encoding_61", 0, 0, {}], ["category_encoding_62", 0, 0, {}], ["category_encoding_63", 0, 0, {}], ["category_encoding_64", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 70, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_20", "inbound_nodes": [[["concatenate_10", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dropout", "config": {"name": "dropout_10", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout_10", "inbound_nodes": [[["dense_20", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_21", "inbound_nodes": [[["dropout_10", 0, 0, {}]]], "shared_object_id": 26}], "input_layers": [["open_to_close_hour", 0, 0], ["response_to_resolved_hour", 0, 0], ["sla", 0, 0], ["product_type", 0, 0], ["brand", 0, 0], ["service_type", 0, 0], ["incident_type", 0, 0]], "output_layers": [["dense_21", 0, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": true, "label_smoothing": 0.0, "axis": -1}, "shared_object_id": 35}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 36}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Custom>Adam", "config": {"name": "Adam", "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "jit_compile": false, "is_legacy_optimizer": false, "learning_rate": 0.0010000000474974513, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "sla", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sla"}}2
�root.layer-1"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "product_type", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "product_type"}}2
�root.layer-2"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "brand", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "brand"}}2
�root.layer-3"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "service_type", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "service_type"}}2
�root.layer-4"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "incident_type", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "incident_type"}}2
�root.layer-5"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "open_to_close_hour", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "open_to_close_hour"}}2
�root.layer-6"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "response_to_resolved_hour", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "response_to_resolved_hour"}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "string_lookup_60", "trainable": true, "expects_training_arg": false, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "StringLookup", "config": {"name": "string_lookup_60", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "inbound_nodes": [[["sla", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�	root.layer_with_weights-1"_tf_keras_layer*�{"name": "string_lookup_61", "trainable": true, "expects_training_arg": false, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "StringLookup", "config": {"name": "string_lookup_61", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "inbound_nodes": [[["product_type", 0, 0, {}]]], "shared_object_id": 8, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�
root.layer_with_weights-2"_tf_keras_layer*�{"name": "string_lookup_62", "trainable": true, "expects_training_arg": false, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "StringLookup", "config": {"name": "string_lookup_62", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "inbound_nodes": [[["brand", 0, 0, {}]]], "shared_object_id": 9, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "string_lookup_63", "trainable": true, "expects_training_arg": false, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "StringLookup", "config": {"name": "string_lookup_63", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "inbound_nodes": [[["service_type", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�root.layer_with_weights-4"_tf_keras_layer*�{"name": "string_lookup_64", "trainable": true, "expects_training_arg": false, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "StringLookup", "config": {"name": "string_lookup_64", "trainable": true, "dtype": "int64", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": false}, "inbound_nodes": [[["incident_type", 0, 0, {}]]], "shared_object_id": 11, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�
�root.layer_with_weights-6"_tf_keras_layer*�{"name": "normalization_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Normalization", "config": {"name": "normalization_25", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "inbound_nodes": [[["response_to_resolved_hour", 0, 0, {}]]], "shared_object_id": 13, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�
�
�
�
�
�
�root.layer_with_weights-7"_tf_keras_layer*�{"name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 70, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_10", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 70}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70]}}2
�
�root.layer_with_weights-8"_tf_keras_layer*�{"name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_10", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 70}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70]}}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 39}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 36}2