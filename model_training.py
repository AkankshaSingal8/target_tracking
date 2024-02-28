model = get_skeleton(params=model_params)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=['mse'])
        # Load pretrained weights
        if hotstart is not None:
            model.load_weights(hotstart)

        model.summary(line_length=80)

    # Train
    history = model.fit(x=training_dataset, validation_data=validation_dataset, epochs=epochs,
                        use_multiprocessing=False, workers=1, max_queue_size=5, verbose=1, callbacks=callbacks)