from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model_trigconv2d import create_trigconv2d_model()
def train(X_train, X_test, y_train, y_test):
  input_shape = X_train.shape[1:]
  num_classes = y_train.shape[1]
  
  model = create_trigconv2d_model(input_shape, num_classes)
  
  model.compile(optimizer = 'AdamW', loss = 'categorical_crossentropy', metrics = ['accuracy'] )
  
  #Stop training if no improvement is seen
  es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, verbose = 1, mode = 'auto', baseline = None, restore_best_weights = True, start_from_epoch=0)
  
  #automatically saves best model weights to disk
  ckpt = ModelCheckpoint("best_model.keras", monitor = "val_loss", save_best_only=True)
  
  history = model.fit(X_train, y_train, batch_size = 32, epochs = 5, verbose = 1, validation_data=(X_test, y_test), callbacks=[es,ckpt])

  return model, history
