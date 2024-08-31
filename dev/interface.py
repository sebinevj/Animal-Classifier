import tkinter as tk
from tkinter import filedialog, Frame, Button, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# image uploader function

# Main method
def main():

    batch_size = 32
    img_height, img_width = 128, 128
    class_names = ['cockroach', 'hummingbird', 'panda', 'shark', 'snake', 'starfish', 'swan', 'tiger', 'wolf', 'wombat']

    train_ds = tf.keras.utils.image_dataset_from_directory(directory="animalsSubset_10classes", labels='inferred', validation_split=0.15, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(directory="animalsSubset_10classes", labels='inferred', validation_split=0.15, subset="validation", seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    num_classes = len(class_names)

    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    epochs=25
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
 
    # defining tkinter object
    app = tk.Tk()
 
    # setting title and basic size to our App
    app.title("Animal Image Viewer")
    app.geometry("560x270")

    frame = Frame(app)
    frame.pack(side=tk.BOTTOM)

    # adding background color to our upload button
    app.option_add("*Label*Background", "white")
    app.option_add("*Button*Background", "lightgreen")
    
    label = tk.Label(app)
    label.pack(pady=10)
    
    # defining our upload buttom
    uploadButton = tk.Button(frame, text="Locate Image", command=lambda:imageUploader(app, label))
    uploadButton.pack(side=tk.LEFT, padx=10, pady=20)

    saveButton = tk.Button(frame, text="Classify Image", command=lambda:imageClassifier(label, model, class_names))
    saveButton.pack(side=tk.RIGHT, padx=10, pady=20)
    
    app.mainloop()


def imageUploader(app, label):
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = filedialog.askopenfilename(filetypes=fileTypes)
 
    # if file is selected
    if len(path):
        img = Image.open(path)
        img = img.resize((200, 200))
        pic = ImageTk.PhotoImage(img)
 
        # re-sizing the app window in order to fit picture
        # and buttom
        app.geometry("560x300")
        label.config(image=pic)
        label.image = pic

    # if no file is selected, then we are displaying below message
    else:
        messagebox.showinfo(title="Warning!", message="No file was uploaded.")


def imageClassifier(label, model, class_names):

    if label and label.image: 

        image = ImageTk.getimage(label.image).convert('RGB')
        image = image.resize((128, 128))

        img_array = tf.expand_dims(image, 0) # Create a batch
        img_array.shape

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        dangerous = False

        if (class_names[np.argmax(score)] == "shark" or class_names[np.argmax(score)] == "tiger" or class_names[np.argmax(score)] == "wombat" or class_names[np.argmax(score)] == "leopard" or class_names[np.argmax(score)] == "wolf"):
            dangerous = True

        if dangerous:
            messagebox.showinfo(title="Classification", message= "This image is most likely a(n) {} with a {:.2f} percent confidence. This animal is likely dangerous."
            .format(class_names[np.argmax(score)], 100 * np.max(score)))
        else:
            messagebox.showinfo(title="Classification", message= "This image is most likely a(n) {} with a {:.2f} percent confidence. This animal is likely not dangerous"
            .format(class_names[np.argmax(score)], 100 * np.max(score)))

        

if __name__ == "__main__":
    main()