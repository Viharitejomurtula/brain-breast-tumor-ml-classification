from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#Stop training if no improvement is seen| min change needed to keep going | min epochs wo improvement|
es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, verbose = 1, mode = 'auto', baseline = None, restore_best_weights = True, start_from_epoch=0)
#automatically saves best model weights to disk(if accuracy drops later best version is saved)| file name is best_model.keras|watches validation loss to make decision on performance
ckpt = ModelCheckpoint("best_model.keras", monitor = "val_loss", save_best_only=True)

history = model.fit(X_train, y_train, batch_size = 32, epochs = 5, verbose = 1, validation_data=(X_test, y_test), callbacks=[es,ckpt])
